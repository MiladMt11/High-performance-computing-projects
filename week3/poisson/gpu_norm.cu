#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

// Following two functions are taken from the slides.
__inline__ __device__
double warpReduceSum(double value) {
    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down_sync(-1, value, i);
    return value;
}

__inline__ __device__
double blockReduceSum(double value) {
    __shared__ double smem[32]; // Max 32 warp sums

    int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;

    if (tid < warpSize)
        smem[tid] = 0;
    __syncthreads();
    value = warpReduceSum(value);
    if (tid % warpSize == 0)
        smem[tid / warpSize] = value;
    __syncthreads();
    if (tid < warpSize)
        value = smem[tid];
    return warpReduceSum(value);
}

__global__ void gpu_norm_kernel(int N, double ***u1, double ***u2, double *norm2_d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    double diff2 = 0.0;
    if (0 < i && 0 < j && 0 < k && i <= N && j <= N && k <= N) {
        double delta = 2.0 / (double)(N + 2);
        double delta2 = delta * delta;

        double x = -1.0 + (i * delta);
        double y = -1.0 + (j * delta);
        double z = -1.0 + (k * delta);
        double sum =
            u1[i - 1][j][k] +
            u1[i][j - 1][k] +
            u1[i][j][k - 1] +
            u1[i + 1][j][k] +
            u1[i][j + 1][k] +
            u1[i][j][k + 1];

        int mask = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z);
        double f = mask * 200.0;
        u2[i][j][k] = (sum + delta2 * f) / 6.0;

        double diff = u1[i][j][k] - u2[i][j][k];
        diff2 = diff * diff;
    }

    int tid = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;

    double diff2sum = blockReduceSum(diff2);
    if (tid == 0) {
        atomicAdd(norm2_d, diff2sum);
    }
}

int gpu_norm(int N, int iter_max, double tolerance, double ***u_h) {
    int iter;

    double ***u1_d, ***u2_d;

    if ( (u1_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u1_d: allocation on gpu failed");
        exit(-1);
    }
    if ( (u2_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u2_d: allocation on gpu failed");
        exit(-1);
    }

    double *norm2_d;
    checkCudaErrors(cudaMalloc((void**)&norm2_d, sizeof(double)));

    transfer_3d(u1_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(u2_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);

    int blocks = (N + 7) / 8;
    dim3 dimGrid(blocks,blocks,blocks);
    dim3 dimBlock(8,8,8);

    for (iter = 0; iter < iter_max; ++iter) {
        // Set norm2_d to zero.
        double norm2 = 0.0;
        checkCudaErrors(cudaMemcpy(norm2_d, &norm2, sizeof(double), cudaMemcpyHostToDevice));

        gpu_norm_kernel<<<dimGrid,dimBlock>>>(N, u1_d, u2_d, norm2_d);
        cudaDeviceSynchronize();

        double ***tmp = u1_d;
        u1_d = u2_d;
        u2_d = tmp;

        // Read value of norm2_d.
        checkCudaErrors(cudaMemcpy(&norm2, norm2_d, sizeof(double), cudaMemcpyDeviceToHost));
        // printf("Norm: %f\n", sqrt(norm2));
        if (norm2 < tolerance * tolerance) break;
    }

    transfer_3d(u_h, u1_d, N+2, N+2, N+2, cudaMemcpyDeviceToHost);

    free_gpu(u1_d);
    free_gpu(u2_d);
    checkCudaErrors(cudaFree(norm2_d));

    return iter;
}
