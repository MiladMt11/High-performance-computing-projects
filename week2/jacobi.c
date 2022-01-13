/* jacobi.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include "alloc3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "omp.h"

// baseline
int jacobi(
    int N,
    int iter_max,
    double tolerance,
    double ****u)
{
    int iter;
    double ***u1, ***u2;
    if ((u2 = d_malloc_3d(N + 2, N + 2, N + 2)) == NULL)
    {
        perror("array u2: allocation failed");
        exit(-1);
    }
    if ((u1 = d_malloc_3d(N + 2, N + 2, N + 2)) == NULL)
    {
        perror("array u1: allocation failed");
        exit(-1);
    }

    double delta = 2.0 / (double)(N + 2);
#ifdef _JACOBI_V0
    // no parallel region
#elif defined(_JACOBI_OMP_V0)
    // no parallel region
#elif defined(_JACOBI_OMP_V1) // parallel for NUMA
#pragma omp parallel for
#elif defined(_JACOBI_OMP_V2)
#pragma omp parallel for schedule(runtime)
#endif
    for (int i = 0; i < N + 2; ++i) // Copy over boundary conditions.
    {
        for (int j = 0; j < N + 2; ++j)
        {
            for (int k = 0; k < N + 2; ++k)
            {
                u1[i][j][k] = (*u)[i][j][k];
                u2[i][j][k] = (*u)[i][j][k];
            }
        }
    }

    for (iter = 0; iter < iter_max; ++iter)
    {
        double norm = 0.0;
#ifdef _JACOBI_V0
        // no parallel region
#elif defined(_JACOBI_OMP_V0)
#pragma omp parallel for reduction(+ \
                                   : norm)
#elif defined(_JACOBI_OMP_V1)
#pragma omp parallel for reduction(+ \
                                   : norm)
#elif defined(_JACOBI_OMP_V2)
#pragma omp parallel for reduction(+ \
                                   : norm) schedule(runtime)
#endif
        for (int i = 1; i < N + 1; ++i)
        {
            double x = -1.0 + (i * delta);
            for (int j = 1; j < N + 1; ++j)
            {
                double y = -1.0 + (j * delta);
                for (int k = 1; k < N + 1; ++k)
                {
                    double z = -1.0 + (k * delta);
                    double sum =
                        u1[i - 1][j][k] +
                        u1[i][j - 1][k] +
                        u1[i][j][k - 1] +
                        u1[i + 1][j][k] +
                        u1[i][j + 1][k] +
                        u1[i][j][k + 1];

#if CHECK_CORRECTNESS
                    double f = 3 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
#else
                    double f = (x <= -0.375 && y <= -0.5 && -(2.0 / 3.0) <= z) ? 200.0 : 0.0;
#endif
                    u2[i][j][k] = (sum + delta * delta * f) / 6.0;

                    double diff = u1[i][j][k] - u2[i][j][k];
                    norm += diff * diff;
                }
            }
        }

        if (norm < tolerance * tolerance)
            break;

        double ***utmp = u2;
        u2 = u1;
        u1 = utmp;
    }

    // copy
    for (int i = 0; i < N + 2; ++i)
    {
        for (int j = 0; j < N + 2; ++j)
        {
            for (int k = 0; k < N + 2; ++k)
            {
                (*u)[i][j][k] = u2[i][j][k];
            }
        }
    }
    free(u1);
    free(u2);

    return iter;
}
