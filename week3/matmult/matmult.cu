#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
#include "cblas.h"
}
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <helper_cuda.h>
#include <omp.h>

#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define max(X, Y) ((X) > (Y) ? (X) : (Y))
#define BLOCK_SIZE 32

extern "C"
{; // just for Ernie's IDE indentation config, please don't remove it :)

void matmult_nat(int m, int n, int k, double* A, double* B, double* C)
{
    // A(m,k) m - # of rows; k - # of columns
    // B(k,n) k - # of rows; n - # of columns
    // C(m,n) m - # of rows; n - # of columns
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (i = 0; i < m; i++) // i - row #
    {
        for (l = 0; l < k; l++) // k - row #
        {
            for (j = 0; j < n; j++) // j - col #
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_lib(int m, int n, int k, double* A, double* B, double* C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
}

void matmult_mkn(int m, int n, int k, double* A, double* B, double* C)
{
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (i = 0; i < m; i++)
    {
        for (l = 0; l < k; l++)
        {
            for (j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_mnk(int m, int n, int k, double* A, double* B, double* C)
{
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            for (l = 0; l < k; l++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_kmn(int m, int n, int k, double* A, double* B, double* C)
{
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (l = 0; l < k; l++)
    {
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_knm(int m, int n, int k, double* A, double* B, double* C)
{
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (l = 0; l < k; l++)
    {
        for (j = 0; j < n; j++)
        {
            for (i = 0; i < m; i++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_nmk(int m, int n, int k, double* A, double* B, double* C)
{
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < m; i++)
        {
            for (l = 0; l < k; l++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_nkm(int m, int n, int k, double* A, double* B, double* C)
{
    int i, j, l;
    memset(C, 0, sizeof(double) * m * n);
    for (j = 0; j < n; j++)
    {
        for (l = 0; l < k; l++)
        {
            for (i = 0; i < m; i++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void matmult_blk(int m, int n, int k, double* A, double* B, double* C, int bs)
{
    int i_block, j_block, l_block;
    memset(C, 0, sizeof(double) * m * n);
    for (i_block = 0; i_block < m; i_block += bs)
    {
        for (l_block = 0; l_block < k; l_block += bs)
        {
            for (j_block = 0; j_block < n; j_block += bs)
            {
                int i, j, l, i_max, j_max, l_max;
                i_max = min(m, i_block + bs);
                j_max = min(n, j_block + bs);
                l_max = min(k, l_block + bs);
                for (i = i_block; i < i_max; i++)
                {
                    for (l = l_block; l < l_max; l++)
                    {
                        for (j = j_block; j < j_max; j++)
                        {
                            C[i * n + j] += A[i * k + l] * B[l * n + j];
                        }
                    }
                }
            }
        }
    }
}

__global__ void gpu1_kernel(int m, int n, int k, double* A, double* B, double* C)
{
    // A(m,k) m - # of rows; k - # of columns
    // B(k,n) k - # of rows; n - # of columns
    // C(m,n) m - # of rows; n - # of columns
    int i, j, l;
    for (i = 0; i < m; i++)
    {
        for (l = 0; l < k; l++)
        {
            for (j = 0; j < n; j++)
            {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

#define TIME_TRANSFER 0

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpu1(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
#ifdef TIME_TRANSFER
    double ts = omp_get_wtime();
#endif
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(double));
#ifdef TIME_TRANSFER
    double te = omp_get_wtime();
    double diff = te - ts;
    printf("CPU => GPU: %f\n", te - ts);
#endif

#ifdef TIME_TRANSFER
    ts = omp_get_wtime();
#endif
    // Launch kernel and synchronize
    gpu1_kernel << <1, 1 >> > (m, n, k, A_d, B_d, C_d);
    cudaDeviceSynchronize();
#ifdef TIME_TRANSFER
    te = omp_get_wtime();
    double runtime = te - ts;
#endif

#ifdef TIME_TRANSFER
    ts = omp_get_wtime();
#endif
    // Copy result back to host
    cudaMemcpy(A_h, A_d, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, k * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free A_d, B_d, C_d
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
#ifdef TIME_TRANSFER
    te = omp_get_wtime();
    printf("GPU => CPU: %f\n", te - ts);
    printf("Total perc of runtime: %f\n", (diff + (ts - ts))/runtime * 100);
#endif
}


__global__ void gpu2_kernel(int m, int n, int k, double* A, double* B, double* C)
{
    // A(m,k) m - # of rows; k - # of columns
    // B(k,n) k - # of rows; n - # of columns
    // C(m,n) m - # of rows; n - # of columns
    int threadRowID, threadColID;
    threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    threadColID = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadColID >= n || threadRowID >= m)
        return;

    double sum = 0.0;
#pragma unroll
    for (int l = 0; l < k; l++)
        sum += A[threadRowID * k + l] * B[l * n + threadColID];

    C[threadRowID * n + threadColID] = sum;
}

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpu2(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(double));

    // Launch kernel and synchronize
    int bs = BLOCK_SIZE; // TODO: if bs too large, doesn't work for small matrices
    int bsx = (m + (bs - 1)) / bs;
    int bsy = (n + (bs - 1)) / bs;
    dim3 dimGrid(bsx, bsy, 1);
    dim3 dimBlock(bs, bs, 1);
    gpu2_kernel << <dimGrid, dimBlock >> > (m, n, k, A_d, B_d, C_d);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(A_h, A_d, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, k * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free A_d, B_d, C_d
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

__global__ void gpu3_kernel(int m, int n, int k, double* A, double* B, double* C)
{
    // A(m,k) m - # of rows; k - # of columns
    // B(k,n) k - # of rows; n - # of columns
    // C(m,n) m - # of rows; n - # of columns
    int threadRowID, threadColID;
    threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    threadColID = blockIdx.y * blockDim.y + threadIdx.y;

    int cid = threadColID * 2;
    if (cid >= n || threadRowID >= m)
        return;

    // TODO: which one is better?

    // int rid = threadRowID * 2;
    // double sum1 = 0.0;
    // double sum2 = 0.0;
    // #pragma unroll
    // for (int l = 0; l < k; l++)
    // {
    //     sum1 += A[rid * k + l] * B[l * n + threadColID];
    //     sum2 += A[(rid + 1) * k + l] * B[l * n + threadColID];
    // }
    // C[rid * n + threadColID] = sum1;
    // C[(rid + 1) * n + threadColID] = sum2;

    if (n - cid == 1)
    {
        double sum = 0.0;
#pragma unroll
        for (int l = 0; l < k; l++)
            sum += A[threadRowID * k + l] * B[l * n + cid];
        C[threadRowID * n + cid] = sum;
        return;
    }

    double sum1 = 0.0;
    double sum2 = 0.0;
#pragma unroll
    for (int l = 0; l < k; l++)
    {
        sum1 += A[threadRowID * k + l] * B[l * n + cid];
        sum2 += A[threadRowID * k + l] * B[l * n + cid + 1];
    }
    C[threadRowID * n + cid] = sum1;
    C[threadRowID * n + cid + 1] = sum2;
}

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpu3(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(double));

    // Launch kernel and synchronize
    int bs = BLOCK_SIZE; // TODO: if bs too large, doesn't work for small matrices
    int bsx = m / bs + 1;
    int bsy = n / 2 / bs + 1;
    dim3 dimGrid(bsx, bsy, 1);
    dim3 dimBlock(bs, bs, 1);
    gpu3_kernel << <dimGrid, dimBlock >> > (m, n, k, A_d, B_d, C_d);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(A_h, A_d, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, k * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free A_d, B_d, C_d
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

#define COMP_N_ELEMENTS 2

__global__ void gpu4_kernel(int m, int n, int k, double* A, double* B, double* C)
{
    // A(m,k) m - # of rows; k - # of columns
    // B(k,n) k - # of rows; n - # of columns
    // C(m,n) m - # of rows; n - # of columns
    int threadRowID, threadColID;
    threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    threadColID = blockIdx.y * blockDim.y + threadIdx.y;

    int cid = threadColID * COMP_N_ELEMENTS;
    if (cid >= n || threadRowID >= m)
        return;

    double sums[COMP_N_ELEMENTS] = { 0 }; int i;
    int diff = n - cid;
    int loop_count = min(COMP_N_ELEMENTS, diff);
#pragma unroll
    for (int l = 0; l < k; l++)
    {
#pragma unroll
        for (i = 0; i < loop_count; i++)
            sums[i] += A[threadRowID * k + l] * B[l * n + (cid + i)];
    }

#pragma unroll
    for (i = 0; i < loop_count; i++)
        C[threadRowID * n + (cid + i)] = sums[i];
}

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpu4(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(double));

    // Launch kernel and synchronize
    int bs = BLOCK_SIZE; // TODO: if bs too large, doesn't work for small matrices
    int bsx = m / bs + 1;
    int bsy = n / COMP_N_ELEMENTS / bs + 1;
    // printf("Grid: (%d,%d)\n",bsx,bsy);
    dim3 dimGrid(bsx, bsy, 1);
    dim3 dimBlock(bs, bs, 1);
    gpu4_kernel << < dimGrid, dimBlock >> > (m, n, k, A_d, B_d, C_d);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(A_h, A_d, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, k * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free A_d, B_d, C_d
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

#define SHARED_BLOCK_SIZE 32

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
__global__ void gpu5_kernel(int m, int n, int k, double* A, double* B, double* C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    double* Csub_ptr = &C[n * SHARED_BLOCK_SIZE * blockRow + SHARED_BLOCK_SIZE * blockCol];

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (k / SHARED_BLOCK_SIZE); ++i) {

        double* Asub = &A[k * SHARED_BLOCK_SIZE * blockRow + SHARED_BLOCK_SIZE * i];

        // Get sub-matrix Asub of A
        // Matrix Asub = GetSubMatrix(A, blockRow, i);

        double* Bsub = &B[n * SHARED_BLOCK_SIZE * i + SHARED_BLOCK_SIZE * blockCol];

        // Get sub-matrix Bsub of B
        // Matrix Bsub = GetSubMatrix(B, i, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double Ash[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
        __shared__ double Bsh[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

        // return A.elements[row * A.stride + col];
        // Ash[row][col] = GetElement(Asub, row, col);
        Ash[row][col] = Asub[row * k + col];
        // Bsh[row][col] = GetElement(Bsub, row, col);
        Bsh[row][col] = Bsub[row * n + col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < SHARED_BLOCK_SIZE; ++e)
            Cvalue += Ash[row][e] * Bsh[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    // SetElement(Csub, row, col, Cvalue);
    Csub_ptr[row * n + col] = Cvalue;
}


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    double* elements;
} Matrix;

// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
    double value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width = SHARED_BLOCK_SIZE;
    Asub.height = SHARED_BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * SHARED_BLOCK_SIZE * row
        + SHARED_BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of SHARED_BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(double);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
        cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(double);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
        cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / SHARED_BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
        __shared__ double Bs[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < SHARED_BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpu5(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(double));

    // Structs
    // Matrix mA, mB, mC;
    // mA.width = k;
    // mA.height = m;
    // mA.stride = m; 
    // mA.elements = A_h;
    // mB.width = n;
    // mB.height = k;
    // mB.stride = k; 
    // mB.elements = B_h;
    // mC.width = n;
    // mC.height = m;
    // mC.stride = m; 
    // mC.elements = C_h;
    // MatMul(mA, mB, mC);

    // 20% faster than using structs
    dim3 dimBlock(SHARED_BLOCK_SIZE, SHARED_BLOCK_SIZE);
    dim3 dimGrid(n / dimBlock.x, m / dimBlock.y);
    gpu5_kernel << <dimGrid, dimBlock >> > (m, n, k, A_d, B_d, C_d);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(A_h, A_d, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h, B_d, k * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_h, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free A_d, B_d, C_d
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpulib(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(double));

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B_d, n, A_d, k, &beta, C_d, n);

    cudaMemcpy(C_h, C_d, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(handle); // destroy CUBLAS context
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

//   Total amount of constant memory:               65536 bytes
//   Total amount of shared memory per block:       49152 bytes
//   Total shared memory per multiprocessor:        167936 bytes

}