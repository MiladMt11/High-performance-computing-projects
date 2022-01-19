//
// File 'example3d.cu' illustrates how to use the functions in 
// alloc3d.h, alloc3d_gpu.h, and transfer3d_gpu.h.
//
#include <stdio.h>
#include <omp.h>
#include "alloc3d.h"
#include "alloc3d_gpu.h"
#include "transfer3d_gpu.h"

#include "omp_jacobi.h"
#include "gpu_seq.h"
#include "gpu_par.h"

int
main(int argc, char *argv[])
{
    if (argc != 6) {
        printf("Usage: %s [cpu_nonorm,cpu_norm,gpu_seq,gpu_par,gpu_par2,gpu_norm] "
               "N max_iters tolerance start_temp\n", argv[0]);
        return 1;
    }

    char *method = argv[1];
    bool needs_gpu = (strcmp(method, "cpu_nonorm") != 0) && (strcmp(method,
                "cpu_norm") != 0);

    int devices = 0;
    if (needs_gpu) {
        cudaError_t err = cudaGetDeviceCount(&devices);
        if (err != cudaSuccess || devices == 0) {
            printf("Error: This machine doesn't have a GPU.\n");
            return 1;
        }
    }

    int N = atoi(argv[2]);
    long nElms = ((long)N) * ((long)N) * ((long)N);
    int iter_max = atoi(argv[3]);
    double tolerance = atof(argv[4]);
    double start_T = atof(argv[5]);




    // Prepare initial conditions of input.
    double ***u_h = NULL;
    if ( (u_h = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }
    for (int i = 0; i < N+2; ++i) {
      for (int j = 0; j < N+2; ++j) {
        for (int k = 0; k < N+2; ++k) {
          u_h[i][j][k] = start_T;
        }
      }
    }
    for (int i = 0; i < N+2; ++i) {
      for (int j = 0; j < N+2; ++j) {
        u_h[i][j][0] = 20.0;
        u_h[i][j][N+1] = 20.0;
        u_h[i][0][j] = 0.0;
        u_h[i][N+1][j] = 20.0;
        u_h[0][i][j] = 20.0;
        u_h[N+1][i][j] = 20.0;
      }
    }





    // Pick appropriate implementation and run it.
    int ret = 0;
    int iters;
    double running_time;
    if (strcmp(method, "cpu_nonorm") == 0) {
        double start = omp_get_wtime();
        iters = cpu_jacobi_nonorm(N, iter_max, u_h);
        double end = omp_get_wtime();
        running_time = end - start;
    }
    else if (strcmp(method, "cpu_norm") == 0) {
        double start = omp_get_wtime();
        iters = cpu_jacobi_norm(N, iter_max, tolerance, u_h);
        double end = omp_get_wtime();
        running_time = end - start;
    }
    else if (strcmp(method, "gpu_seq") == 0) {
        double start = omp_get_wtime();
        iters = gpu_seq(N, iter_max, u_h);
        double end = omp_get_wtime();
        running_time = end - start;
    }
    else if (strcmp(method, "gpu_par") == 0) {
        double start = omp_get_wtime();
        iters = gpu_par(N, iter_max, u_h);
        double end = omp_get_wtime();
        running_time = end - start;
    }
    else if (strcmp(method, "gpu_par2") == 0) {
        printf("%s is not yet implemented.\n", method);
        ret = 1;
    }
    else if (strcmp(method, "gpu_norm") == 0) {
        printf("%s is not yet implemented.\n", method);
        ret = 1;
    } else {
        printf("No such implementation %s.\n", method);
        ret = 1;
    }

    double bytes_alloc = sizeof(double) * (double) nElms;
    double flops_per_sec = (8 * iters * (double) nElms) / running_time;

    printf("%s %d %d %f %f %f\n", method, N, iters, bytes_alloc / 1e3,
            flops_per_sec / 1e6, running_time);

    free(u_h);
    return ret;

    /*
    const int N = 200;            // Dimension N x N x N.
    const long nElms = N * N * N; // Number of elements.

    double 	***u_h = NULL;
    double 	***u_d0 = NULL;
    double 	***u_d1 = NULL;

    // Allocate 3d array in host memory.
    if ( (u_h = d_malloc_3d(N, N, N)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // Allocate 3d array of half size in device 0 memory.
    if ( (u_d0 = d_malloc_3d_gpu(N / 2, N, N)) == NULL ) {
        perror("array u_d0: allocation on gpu failed");
        exit(-1);
    }

    // Allocate 3d array of half size in device 1 memory.
    if ( (u_d1 = d_malloc_3d_gpu(N / 2, N, N)) == NULL ) {
        perror("array u_d1: allocation on gpu failed");
        exit(-1);
    }

    // Transfer top part to device 0.
    transfer_3d_from_1d(u_d0, u_h[0][0], N / 2, N, N, cudaMemcpyHostToDevice);

    // Transfer bottom part to device 1.
    transfer_3d_from_1d(u_d1, u_h[0][0] + nElms / 2, N / 2, N, N, cudaMemcpyHostToDevice);

    // ... compute ...

    // ... transfer back ...

    // Clean up.
    free(u_h);
    free_gpu(u_d0);
    free_gpu(u_d1);

    printf("Done\n");
    */
}
