#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void gpu_par_f_kernel(int N, double ***u1, double ***u2, double ***f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (0 < i && 0 < j && 0 < k && i <= N && j <= N && k <= N) {
        double delta = 2.0 / (double)(N + 2);
        double delta2 = delta * delta;

        double sum =
            u1[i - 1][j][k] +
            u1[i][j - 1][k] +
            u1[i][j][k - 1] +
            u1[i + 1][j][k] +
            u1[i][j + 1][k] +
            u1[i][j][k + 1];

        u2[i][j][k] = (sum + delta2 * f[i][j][k]) / 6.0;
    }
}

int gpu_par_f(int N, int iter_max, double ***u_h, double ***f_h) {
    int iter;

    double ***u1_d, ***u2_d, ***f_d;

    if ( (u1_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u1_d: allocation on gpu failed");
        exit(-1);
    }
    if ( (u2_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array u2_d: allocation on gpu failed");
        exit(-1);
    }
    if ( (f_d = d_malloc_3d_gpu(N+2, N+2, N+2)) == NULL ) {
        perror("array f_d: allocation on gpu failed");
        exit(-1);
    }

    transfer_3d(u1_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(u2_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(f_d, f_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);

    int blocks = (N + 7) / 8;
    dim3 dimGrid(blocks,blocks,blocks);
    dim3 dimBlock(8,8,8);

    for (iter = 0; iter < iter_max; ++iter) {
        gpu_par_f_kernel<<<dimGrid,dimBlock>>>(N, u1_d, u2_d, f_d);
        cudaDeviceSynchronize();

        double ***tmp = u1_d;
        u1_d = u2_d;
        u2_d = tmp;
    }

    transfer_3d(u_h, u1_d, N+2, N+2, N+2, cudaMemcpyDeviceToHost);

    free_gpu(u1_d);
    free_gpu(u2_d);
    free_gpu(f_d);

    return iter;
}
