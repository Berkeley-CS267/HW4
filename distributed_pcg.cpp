#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <utility>

#include <Eigen/Sparse>

class Matrix{
  public:
    typedef std::pair<int, int> N2;
  
    std::map<N2, double> data;
    int nbrow;
    int nbcol;

    Matrix(const int& nr = 0, const int& nc = 0): nbrow(nr), nbcol(nc) {
      for (int i = 0; i < nc; ++i) {
        data[std::make_pair(i, i)] = 2.0;
        if (i - 1 >= 0) data[std::make_pair(i, i - 1)] = -1.0;
        if (i + 1 < nc) data[std::make_pair(i, i + 1)] = -1.0;
      }
    }; 
  
    int NbRow() const {return nbrow;}
    int NbCol() const {return nbcol;}
  
    // matrix-vector product with vector xi
    std::vector<double> operator*(const std::vector<double>& xi) const {
      std::vector<double> b(NbRow(), 0.);
      for(auto it = data.begin(); it != data.end(); ++it){
        int j = (it->first).first;
        int k = (it->first).second; 
        double Mjk = it->second;
        b[j] += Mjk * xi[k];
      }
  
      return b;
    }
};
  
// scalar product (u, v)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  double sp = 0.;
  for(int j = 0; j < u.size(); j++)
    sp += u[j] * v[j];
  return sp; 
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  std::vector<double> w = u;
  for(int j = 0; j < u.size(); j++)
    w[j] += v[j];
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j = 0; j < w.size(); j++) 
    w[j] = a * u[j];
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  for(int j = 0; j < u.size(); j++)
    u[j] += v[j];
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u){
  Eigen::VectorXd b(u.size());
  for (int i = 0; i < u.size(); i++) 
    b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b);
  std::vector<double> x(u.size());
  for (int i = 0; i < u.size(); i++) 
    x[i] = xe[i];
  return x;
}

Matrix A;

/* N is the size of the matrix, and n is the number of rows assigned per rank.
 * It is your responsibility to generate the input matrix, assuming the ranks are 
 * partitioned rowwise.
 * The input matrix is L, where L represents a discretized 1D Possion's equation.
 * That is to say L has 2s on its diagonal and -1s on it super/sub-diagonals.
 * See the constructor of the Matrix structure as an example.
 * The constructor of CG_Solver will not be included in the timing result.
 * Note that the starter code only works for 1 rank and it is not efficient.
 */
CG_Solver::CG_Solver(const int& n, const int& N) {
  A = Matrix(n, N);
}

/* The preconditioned conjugate gradient method solving Ax = b with tolerance tol.
 * This is the function being evalauted for performance.
 * Note that the starter code only works for 1 rank and it is not efficient.
 */
void CG_Solver::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int n = A.NbRow();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for(auto it = A.data.begin(); it != A.data.end(); ++it){
    int j = (it->first).first;
    int k = (it->first).second;
    coefficients.push_back(Eigen::Triplet<double>(j, k, it -> second)); 
  }

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n, n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  const double epsilon = tol * std::sqrt((b, b));
  x.assign(b.size(), 0.);
  std::vector<double> r = b, z = prec(P, b), p = z;
  double alpha = 0., beta = 0.;
  double res = std::sqrt((r, r));

  int num_it = 0;
  
  while(res >= epsilon) {
    alpha = (r, z) / (p, A * p);
    x += (+alpha) * p; 
    r += (-alpha) * (A * p);
    z = prec(P, r);
    beta = (r, z) / (alpha * (p, A * p)); 
    p = z + beta * p;    
    res = std::sqrt((r, r));
    
    num_it++;
    if (rank == 0 && !(num_it % 1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << res << "\n";
    }
  }
 }