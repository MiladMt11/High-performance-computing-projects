#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

__global__ void gpu_par2_kernel0(int N, int N1, double ***u1, double ***u2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (0 < i && 0 < j && 0 < k && i < N1 && j <= N && k <= N) {
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
    }
}

__global__ void gpu_par2_kernel1(int N, int N2, double ***u1, double ***u2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (0 < i && 0 < j && 0 < k && i < N2 && j <= N && k <= N) {
        double delta = 2.0 / (double)(N + 2);
        double delta2 = delta * delta;

        //double x = -1.0 + ((N1 + i) * delta);
        //double y = -1.0 + (j * delta);
        //double z = -1.0 + (k * delta);
        double sum =
            u1[i - 1][j][k] +
            u1[i][j - 1][k] +
            u1[i][j][k - 1] +
            u1[i + 1][j][k] +
            u1[i][j + 1][k] +
            u1[i][j][k + 1];

        // f is always zero in this kernel
        //double f = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z) ? 200.0 : 0.0;
        u2[i][j][k] = (sum /*+ delta2 * f*/) / 6.0;
    }
}

int gpu_par2(int N, int iter_max, double ***u_h) {
    int iter;

    int N1 = (N+2) / 2;
    int N2 = (N+2) - N1;

    cudaSetDevice(0);
    double ***u1_d0, ***u2_d0;
    if ( (u1_d0 = d_malloc_3d_gpu(N1+1, N+2, N+2)) == NULL ) {
        perror("array u1_d0: allocation on gpu failed");
        exit(-1);
    }
    if ( (u2_d0 = d_malloc_3d_gpu(N1+1, N+2, N+2)) == NULL ) {
        perror("array u2_d0: allocation on gpu failed");
        exit(-1);
    }

    cudaSetDevice(1);
    double ***u1_d1, ***u2_d1;
    if ( (u1_d1 = d_malloc_3d_gpu(N2+1, N+2, N+2)) == NULL ) {
        perror("array u1_d1: allocation on gpu failed");
        exit(-1);
    }
    if ( (u2_d1 = d_malloc_3d_gpu(N2+1, N+2, N+2)) == NULL ) {
        perror("array u2_d1: allocation on gpu failed");
        exit(-1);
    }

    // Transfer data into matrices.
    cudaSetDevice(0);
    transfer_3d_from_1d(u1_d0, u_h[0][0], N1+1, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(u2_d0, u_h[0][0], N1+1, N+2, N+2, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    transfer_3d_from_1d(u1_d1, u_h[N1-1][0], N2+1, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d_from_1d(u2_d1, u_h[N1-1][0], N2+1, N+2, N+2, cudaMemcpyHostToDevice);

    // Set up pointers for cross-GPU access.
    cudaSetDevice(0);
    // u1_d0[N1] = u1_d1[1]
    double **u1_d1_last = (double **) u1_d1 + (N2+1) + 1 * (N+2); // u1_d1[1]
    checkCudaErrors(cudaMemcpy((void*) (u1_d0 + N2), (void*) &u1_d1_last, sizeof(double**), cudaMemcpyHostToDevice));
    // u2_d0[N1] = u2_d1[1]
    double **u2_d1_last = (double **) u2_d1 + (N2+1) + 1 * (N+2); // u2_d1[1]
    checkCudaErrors(cudaMemcpy((void*) (u2_d0 + N2), (void*) &u2_d1_last, sizeof(double**), cudaMemcpyHostToDevice));

    cudaSetDevice(1);
    // u1_d1[0] = u1_d0[N1-1]
    double **u1_d0_last = (double **) u1_d0 + (N1+1) + (N1-1) * (N+2); // u1_d0[N1-1]
    checkCudaErrors(cudaMemcpy((void*) u1_d1, (void*) &u1_d0_last, sizeof(double**), cudaMemcpyHostToDevice));
    // u2_d1[0] = u2_d0[N1-1]
    double **u2_d0_last = (double **) u2_d0 + (N1+1) + (N1-1) * (N+2); // u2_d0[N1-1]
    checkCudaErrors(cudaMemcpy((void*) u2_d1, (void*) &u2_d0_last, sizeof(double**), cudaMemcpyHostToDevice));

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    int blocks = (N + 8) / 8;
    int blocks0 = (N1 + 8) / 8;
    int blocks1 = (N2 + 8) / 8;
    dim3 dimGrid0(blocks0,blocks,blocks);
    dim3 dimGrid1(blocks1,blocks,blocks);
    dim3 dimBlock(8,8,8);

    cudaSetDevice(0);
    for (iter = 0; iter < iter_max; ++iter) {
        gpu_par2_kernel0<<<dimGrid0,dimBlock>>>(N, N1, u1_d0, u2_d0);
        cudaSetDevice(1);
        gpu_par2_kernel1<<<dimGrid1,dimBlock>>>(N, N2, u1_d1, u2_d1);

        checkCudaErrors(cudaDeviceSynchronize());
        cudaSetDevice(0);
        checkCudaErrors(cudaDeviceSynchronize());

        double ***tmp = u1_d0;
        u1_d0 = u2_d0;
        u2_d0 = tmp;

        tmp = u1_d1;
        u1_d1 = u2_d1;
        u2_d1 = tmp;
    }

    // Note: The copy from device 1 includes the ghost strip with dummy values.
    // They are overwritten in the second copy, which avoids copying the dummy
    // strip.
    cudaSetDevice(1);
    transfer_3d_to_1d(u_h[N1-1][0], u1_d1, N2+1, N+2, N+2, cudaMemcpyDeviceToHost);

    cudaSetDevice(0);
    // Avoid copying over the dummy strip by specifying a shorter length
    // than in transfer_3d_to_1d.
    checkCudaErrors(cudaMemcpy(
                (double *) u_h[0][0],
                (double *) u1_d0 + (N1+1) + (N1+1)*(N+2),
                (N+2)*(N+2)*N1 * sizeof(double),
                cudaMemcpyDeviceToHost));

    free_gpu(u1_d0);
    free_gpu(u2_d0);
    cudaSetDevice(1);
    free_gpu(u1_d1);
    free_gpu(u2_d1);

    cudaSetDevice(0);
    return iter;
}
