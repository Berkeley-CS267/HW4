# Distributed Preconditioned Conjugate Gradient (PCG)

This project aims to implement a distributed Preconditioned Conjugate Gradient (PCG) solver using MPI.

## Building the Code

1. **Load CMake:**
    ```sh
    module load cmake
    ```

2.  **Create a build directory:**
    ```sh
    mkdir build
    cd build
    ```

3.  **Configure the project using CMake:**
    ```sh
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```

4.  **Compile the code:**
    ```sh
    make
    ```
    This will create an executable named `pcg` in the build directory.

## The problem
The goal is to solve $Ax = b$ where $A = L$ is an $N \times N$ s.p.d. matrix with $L$ representing the discretized 1D Poisson's equation. The right hand side $b$ is all 1s and the preconditioned conjugate gradient starts with an initial guess $x$ of all 0s. You can only modify the `distributed_pcg.cpp` file.

## Running the Code

```sh
salloc -N 1 -A mp309 -t 10:00 --qos=interactive -C cpu srun -N 1 --ntasks-per-node <number of tasks> ./pcg -N <size of the matrix>
```
We will use only one node in this homework. Please change the number of tasks and the size of the matrix appropriately for your scaling experiments. For simplicity, we assume that the size of the matrix is a multiple of the number of tasks used. Note that the starter code only works for one task and it is not efficient.

## Submitting the homework
Ensure that your write-up is located in your source directory, next to distributed_pcg.cpp. It should be named cs267XY_hw4.pdf with XY being your group ID.
```sh
cmake -DGROUP_NAME="YourGroupID XY" ..
make package
```
This second command will fail if the PDF is not present. Confirm that it worked using the following command. You should see output like:

```sh
demmel@perlmutter:~/HW4/build> tar tfz cs267XY_hw4.tar.gz 
cs267XY_hw4/cs267XY_hw4.pdf 
cs267XY_hw4/distributed_pcg.cpp
```

Then submit your cs267XY_hw4.tar.gz through bCourses.