#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <numeric>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class MapMatrix {
public:
  // typedef std::pair<int,int>   N2;

  // std::map<N2,double>  data;

  // using m, n for the spmat structure to differentiate for now 
  int rows, cols; // m=num rows, n = num columns 

  std::vector<double> V; // nonzero values. size (num nonzeros) can call this 'data'
  std::vector<double> col_idxs; // column idx of nonzeros, size num nonzeros
  std::vector<double> row_ptrs; // length num rows + 1. It is the index in V where the row starts 

public:
  MapMatrix(const int& nr, const int& nc):
    rows(nr), cols(nc) {row_ptrs.resize(nr + 1, 0);};

  MapMatrix(const MapMatrix& m): 
    // nbrow(m.nbrow), nbcol(m.nbcol), data(m.data) {}; 
    rows(m.mrows()), cols(m.ncols()), V(m.V), col_idxs(m.col_idxs), row_ptrs(m.row_ptrs) {}; 
  
  MapMatrix& operator=(const MapMatrix& m){ 
    if(this!=&m){
      rows=m.rows;
      cols=m.cols;
      // data=m.data;
      V = m.V; 
      col_idxs = m.col_idxs; 
      row_ptrs = m.row_ptrs; 
    }   
    return *this; 
  }
  int mrows() const {return rows;}
  int ncols() const {return cols;}

  void insert(int i, int j, double val) {
    V.push_back(val); 
    col_idxs.push_back(j);
     
    // if (i >=row_ptrs.size()-1) { // check if we are adding a new row 
    //   // row_ptrs is the size of num nonzeros - 1 
    //   row_ptrs.push_back(V.size()-1);
    // }
    for (int r = i + 1; r <= rows; ++r) {
        ++row_ptrs[r];
    }
    
  }


  double operator()(const int& j, const int& k) const {
    // find (i,j) ith row, jth col
    for (int i = row_ptrs[j]; i<row_ptrs[j+1]; i++) { 
      if (col_idxs[i] == k) {
        return V[i];
      }
    }

    return 0.0;
  }

  void print() const{
    std::cout << "nonzeros: " << V.size() << std::endl;
    std::cout << "rows: " << rows << std::endl; 
    std::cout << "row_ptrs size: " << row_ptrs.size() << std::endl;
    std::cout << "cols " << cols << std::endl;
    std::cout << "col idxs: " << col_idxs.size() << std::endl;
    int idx = 0; 
    for (int i = 0; i<rows; i++) {
        for (int j=0; j<cols;j++) {
            if (idx < row_ptrs[i+1] && col_idxs[idx]==j) {
              // if current col is column index at idx , found the nonzero to print 
              std::cout << V[idx] << "\t";
              idx++;
            } 
            else {
              std::cout << "0\t";
            }
        }
        std::cout << std::endl;
    }
  }
  

  // parallel matrix-vector product with distributed vector xi
  std::vector<double> operator*(const std::vector<double>& xi) const {
    
    // A*x where A is mxn and x is nx1 
    // CSR format has length m+1 (last element is NNZ) from wikipedia pg on CSR 
    std::vector<double> result(row_ptrs.size() - 1, 0.0); 
    // std::vector<double> result(V.size(), 0.0);
    // loop over rows, for each row, do nonzeros 
    for (int i=0; i < rows; i++) {
      // this is k=row_ptrs[i] to row_ptrs[i+1] - 1
      for (int k=row_ptrs[i]; k < row_ptrs[i+1]; k++) {
        result[i] += V[k] * xi[col_idxs[k]]; 
        // k will index into the values list because row_ptrs holds the indices of the values in V in each row 
        // it also works for col indicies (k-th nonzero element)
      }
    }
    return result; 

  }
};

#include <cmath>

// parallel scalar product (u,v) (u and v are distributed)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  double sp=0.;
  for(int j=0; j<u.size(); j++){sp+=u[j]*v[j];}

  return sp; 
}

// norm of a vector u
double Norm(const std::vector<double>& u) { 
  return sqrt((u,u));
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  std::vector<double> w=u;
  for(int j=0; j<u.size(); j++){w[j]+=v[j];}
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j=0; j<w.size(); j++){w[j]=a*u[j];}
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  for(int j=0; j<u.size(); j++){u[j]+=v[j];}
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u){
  Eigen::VectorXd b(u.size());
  for (int i=0; i<u.size(); i++) b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b); // solves Px=b (=xe)
  std::vector<double> x(u.size());
  for (int i=0; i<u.size(); i++) x[i] = xe[i];
  return x;
}

// distributed conjugate gradient
void CG(const MapMatrix& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        double tol=1e-6) {

  assert(b.size() == A.mrows());
  x.resize(b.size(),0.0);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int n = A.mrows();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for (int i=0; i < n; i++) {
    for (int k=A.row_ptrs[i]; k <A.row_ptrs[i+1]; k++) {
      int j = A.col_idxs[k]; 
      if (j>= 0 && j < n) coefficients.push_back(Eigen::Triplet<double>(i,j,A.V[k])); 
    }
  }

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n,n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  std::vector<double> r=b, z=prec(P,r), p=z, Ap=A*p;
  double np2=(p,Ap), alpha=0.,beta=0.;
  double nr = sqrt((z,r));
  double epsilon = tol*nr;

  std::vector<double> res = A*x;
  res += (-1)*b;
  
  double rres = sqrt((res,res));

  int num_it = 0;
  while(rres>1e-5) {
    alpha = (nr*nr)/(np2);
    x += (+alpha)*p; 
    r += (-alpha)*Ap;
    z = prec(P,r);
    nr = sqrt((z,r));
    beta = (nr*nr)/(alpha*np2); 
    p = z+beta*p;    
    Ap=A*p;
    np2=(p,Ap);

    rres = sqrt((r,r));

    num_it++;
    if(rank == 0 && !(num_it%1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << rres     << "\n";
    }
  }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}

int main(int argc, char* argv[]) {
//   MapMatrix A(3,3); 
//   A.insert(0,0, 1.0);
//   A.insert(0,1, 2.0);
//   A.insert(1,1, 3.0);
//   A.insert(1,2, 4.0);
//   A.insert(2,2, 5.0);
//   std::cout << "(0,0): " << A(0,0) << std::endl;
//   std::cout << "(0,1): " << A(0,1) << std::endl;
//   std::cout << "(1,1): " << A(1,1) << std::endl;
//   std::cout << "(1,2): " << A(1,2) << std::endl;
//   std::cout << "(2,2): " << A(2,2) << std::endl;
//   A.print(); 
//   for (const double& it: Ax) {
//     std::cout << it << "\t";
//   }
  


  MPI_Init(&argc, &argv); // Initialize the MPI environment
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
        return 0;
    }

  int N = find_int_arg(argc, argv, "-N", 100000); // global size

  assert(N%size == 0);
  int n = N/size; // number of local rows

  // row-distributed matrix
  MapMatrix A(n,N);

  int offset = n*rank;

  // local rows of the 1D Laplacian matrix; local column indices start at -1 for rank > 0
  for (int i=0; i<n; i++) {
    A.Assign(i,i)=2.0;
    if (offset + i - 1 >= 0) A.Assign(i,i - 1) = -1;
    if (offset + i + 1 < N)  A.Assign(i,i + 1) = -1;
    if (offset + i + N < N) A.Assign(i, i + N) = -1;
    if (offset + i - N >= 0) A.Assign(i, i - N) = -1;
  }

  // initial guess
  std::vector<double> x(n,0);

  // right-hand side
  std::vector<double> b(n,1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  CG(A,b,x);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << "wall time for CG: " << MPI_Wtime()-time << std::endl;

  std::vector<double> r = A*x + (-1)*b;

  double err = Norm(r)/Norm(b);
  if (rank == 0) std::cout << "|Ax-b|/|b| = " << err << std::endl;

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}
