#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cblas.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <helper_cuda.h>

#define min(X, Y) ((X) < (Y) ? (X) : (Y))

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

// The matrix sizes of A and B are m×k and k×n, respectively, so that C has size m×n
void matmult_gpu1(int m, int n, int k, double* A_h, double* B_h, double* C_h)
{
    // Allocate A_d, B_d, C_d
    double* A_d, * B_d, * C_d;
    cudaMalloc((void**)&A_d, m * k * sizeof(double));
    cudaMalloc((void**)&B_d, k * n * sizeof(double));
    cudaMalloc((void**)&C_d, m * n * sizeof(double));

    // Transfer data
    cudaMemcpy(A_d, A_h, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(C_d, 0, m * n * sizeof(int));

    // Launch kernel and synchronize
    gpu1_kernel << <1, 1 >> > (m, n, k, A_d, B_d, C_d);
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

#define BLOCK_SIZE 8

__global__ void gpu2_kernel(int m, int n, int k, double* A, double* B, double* C, int bsx, int bsy)
{
    // A(m,k) m - # of rows; k - # of columns
    // B(k,n) k - # of rows; n - # of columns
    // C(m,n) m - # of rows; n - # of columns

    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int j = blockIdx.y * blockDim.y + threadIdx.y;

    int threadRowID, threadColID;
    threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    threadColID = blockIdx.y * blockDim.y + threadIdx.y;

    // /* ------------------------------------
    //    Print the thread's 2 dim grid ID
    //    ------------------------------------ */
    // printf("Blk: (%d,%d) Thread: (%d,%d) -> Row/Col = (%d,%d)\n",
    //     blockIdx.x, blockIdx.y,
    //     threadIdx.x, threadIdx.y,
    //     threadRowID, threadColID);

    // if (i * j >= m * n)
    //     return;
    if (threadColID >= n || threadRowID >= m)
        return;

    // if (i * j > 3000)
    // {
    // }

    for (int l = 0; l < k; l++)
        C[threadRowID * n + threadColID] += A[threadRowID * k + l] * B[l * n + threadColID];

    // int i, j, l;
    // for (i = 0; i < m; i++)
    // {
    //     for (l = 0; l < k; l++)
    //     {
    //         for (j = 0; j < n; j++)
    //         {
    //             C[i * n + j] += A[i * k + l] * B[l * n + j];
    //         }
    //     }
    // }
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
    cudaMemset(C_d, 0, m * n * sizeof(int));

    // Launch kernel and synchronize
    int bs = BLOCK_SIZE; // TODO: if bs too large, doesn't work for small matrices
    int bsx = (m + (bs - 1)) / bs;
    int bsy = (n + (bs - 1)) / bs;
    dim3 dimGrid(bsx, bsy, 1);
    dim3 dimBlock(bs, bs, 1);
    gpu2_kernel << <dimGrid, dimBlock >> > (m, n, k, A_d, B_d, C_d, bsx, bsy);
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

}