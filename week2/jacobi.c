/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include "alloc3d.h"
#include <stdio.h>
#include <stdlib.h>

// baseline
double ***jacobi(
    int N,
    int iter_max,
    double tolerance,
    double ***u
) {
    double delta = 2.0 / (double) N;

    double ***u1, ***u2;
    u1 = u;
    if ( (u2 = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u2: allocation failed");
        exit(-1);
    }

    // Copy over boundary conditions.
    for (int i = 0; i < N+2; ++i) {
        for (int j = 0; j < N+2; ++j) {
            for (int k = 0; k < N+2; ++k) {
                u2[i][j][k] = u[i][j][k];
            }
        }
    }

    for (int iter = 0; iter < iter_max; ++iter) {
        for (int i = 1; i < N+1; ++i) {
            double x = -1.0 + (i * delta);
            for (int j = 1; j < N+1; ++j) {
                double y = -1.0 + (j * delta);
                for (int k = 1; k < N+1; ++k) {
                    double z = -1.0 + (k * delta);
                    double sum =
                        u1[i-1][j][k] +
                        u1[i][j-1][k] +
                        u1[i][j][k-1] +
                        u1[i+1][j][k] +
                        u1[i][j+1][k] +
                        u1[i][j][k+1];

#if CHECK_CORRECTNESS
                    double f = 3*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
#else
                    double f = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z) ? 200.0 : 0.0;
#endif
                    u2[i][j][k] = (sum + delta*delta * f) / 6.0;
                }
            }
        }

        double sum2 = 0.0;
        for (int i = 1; i < N+1; ++i) {
            for (int j = 1; j < N+1; ++j) {
                for (int k = 1; k < N+1; ++k) {
                    double diff = u1[i][j][k] - u2[i][j][k];
                    sum2 += diff * diff;
                }
            }
        }

        if (sum2 * sum2 < tolerance) {
            break;
        }

        double ***utmp = u2;
        u2 = u1;
        u1 = utmp;
    }

    free(u1);
    return u2;
}

