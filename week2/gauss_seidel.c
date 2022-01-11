/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void gauss_seidel(
    int N,
    int iter_max,
    double tolerance,
    double ***u
) {
    double delta = 2.0 / (double) N;

    for (int iter = 0; iter < iter_max; ++iter) {
        double norm2 = 0.0;
        for (int i = 1; i < N+1; ++i) {
            double x = -1.0 + (i * delta);
            for (int j = 1; j < N+1; ++j) {
                double y = -1.0 + (j * delta);
                for (int k = 1; k < N+1; ++k) {
                    double z = -1.0 + (k * delta);
                    double sum =
                        u[i-1][j][k] +
                        u[i][j-1][k] +
                        u[i][j][k-1] +
                        u[i+1][j][k] +
                        u[i][j+1][k] +
                        u[i][j][k+1];

                    double f = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z) ? 200.0 : 0.0;

                    double old = u[i][j][k];
                    double new = (sum + delta*delta * f) / 6.0;
                    double diff = old - new;
                    norm2 += diff * diff;

                    u[i][j][k] = new;
                }
            }
        }

        printf("%d %f\n", iter, norm2);
        if (norm2 * norm2 < tolerance) {
            break;
        }
    }

}

