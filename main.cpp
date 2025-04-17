#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include "common.h"

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], option) == 0) {
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
  MPI_Init(&argc, &argv); // Initialize the MPI environment
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  if (find_arg_idx(argc, argv, "-h") >= 0) {
      std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
      return 0;
  }

  int N = find_int_arg(argc, argv, "-N", 1 << 20); // global size

  assert(N % size == 0);
  int n = N / size; // number of local rows

  // generate L + I
  CG_Solver cg(n, N);

  // initial guess
  std::vector<double> x(n, 0);

  // right-hand side
  std::vector<double> b(n, 1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  cg.solve(b, x, 1e-8);

  MPI_Barrier(MPI_COMM_WORLD);

  // Do not modify this line, use for grading
  if (rank == 0) {
    std::cout << "Time for CG of size " << N << " with " 
              << size << " rank(s): " << MPI_Wtime() - time 
              << " seconds." << std::endl;
  }
  
  std::vector<double> global_x;
  
  if (rank == 0)
    global_x.resize(N);

  MPI_Gather(x.data(), n, MPI_DOUBLE, global_x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    double r_square = 0;
    for (int i = 0; i < N; ++i) {
      double r = global_x[i] * 2;
      if (i > 0)  r -= global_x[i - 1];
      if (i + 1 < N)  r -= global_x[i + 1];
      r_square += (r - 1) * (r - 1);
    }
    std::cout << "|Ax - b| / |b| = " << std::sqrt(r_square) / std::sqrt(N) << std::endl;
  }

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}