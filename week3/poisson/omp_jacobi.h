#ifndef _OMP_JACOBI_H
#define _OMP_JACOBI_H

int cpu_jacobi_norm(
    int N,
    int iter_max,
    double tolerance,
    double ***u
);

int cpu_jacobi_nonorm(
    int N,
    int iter_max,
    double tolerance,
    double ***u
);

#endif
