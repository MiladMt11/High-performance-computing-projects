/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

int gauss_seidel(
    int N,
    int iter_max,
    double tolerance,
    double ***u
);

#endif
