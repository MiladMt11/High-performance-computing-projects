/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int gauss_seidel(
    int N,
    int iter_max,
    double tolerance,
    double ***u
) {
    int iter;
    double delta = 2.0 / (double) N;

    for (iter = 0; iter < iter_max; ++iter) {
        double norm2 = 0.0;
        #ifdef _GAUSS_OMP
        #pragma omp parallel for ordered(2) schedule(static,1) reduction(+: norm2)
        #endif
        for (int i = 1; i < N+1; ++i) {
            for (int j = 1; j < N+1; ++j) {
                double x = -1.0 + (i * delta);
                double y = -1.0 + (j * delta);
                #ifdef _GAUSS_OMP
                #pragma omp ordered depend(sink: i-1,j) depend(sink: i,j-1)
                #endif
                for (int k = 1; k < N+1; ++k) {
                    double z = -1.0 + (k * delta);
                    double sum =
                        u[i-1][j][k] +
                        u[i][j-1][k] +
                        u[i][j][k-1] +
                        u[i+1][j][k] +
                        u[i][j+1][k] +
                        u[i][j][k+1];
                    #if CHECK_CORRECTNESS
                    double f = 3*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
                    #else
                    double f = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z) ? 200.0 : 0.0;
                    #endif
                    double old = u[i][j][k];
                    double new = (sum + delta*delta * f) / 6.0;
                    double diff = old - new;
                    norm2 += diff * diff;

                    u[i][j][k] = new;
                }
                #ifdef _GAUSS_OMP
                #pragma omp ordered depend(source)
                #endif
            }
        }

        if (norm2 < tolerance * tolerance) {
            break;
        }
    }

    return iter;
}

