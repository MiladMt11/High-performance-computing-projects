#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void gpu_seq_kernel(int N, double ***u1, double ***u2) {
    double delta = 2.0 / (double)(N + 2);
    double delta2 = delta * delta;

    for (int i = 1; i < N + 1; ++i) {
        double x = -1.0 + (i * delta);
        for (int j = 1; j < N + 1; ++j) {
            double y = -1.0 + (j * delta);
            for (int k = 1; k < N + 1; ++k) {
                double z = -1.0 + (k * delta);
                double sum =
                    u1[i - 1][j][k] +
                    u1[i][j - 1][k] +
                    u1[i][j][k - 1] +
                    u1[i + 1][j][k] +
                    u1[i][j + 1][k] +
                    u1[i][j][k + 1];

                double f = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z) ? 200.0 : 0.0;
                u2[i][j][k] = (sum + delta2 * f) / 6.0;
            }
        }
    }
}

int gpu_seq(int N, int iter_max, double ***u_h) {
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

    transfer_3d(u1_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);
    transfer_3d(u2_d, u_h, N+2, N+2, N+2, cudaMemcpyHostToDevice);

    for (iter = 0; iter+1 < iter_max; iter += 2) {
        gpu_seq_kernel<<<1,1>>>(N, u1_d, u2_d);
        cudaDeviceSynchronize();
        gpu_seq_kernel<<<1,1>>>(N, u2_d, u1_d);
        cudaDeviceSynchronize();
    }

    double ***ulast_d = u1_d;
    if (iter < iter_max) {
        gpu_seq_kernel<<<1,1>>>(N, u1_d, u2_d);
        cudaDeviceSynchronize();

        ++iter;
        ulast_d = u2_d;
    }

    transfer_3d(u_h, ulast_d, N+2, N+2, N+2, cudaMemcpyDeviceToHost);

    free_gpu(u1_d);
    free_gpu(u2_d);

    return iter;
}
