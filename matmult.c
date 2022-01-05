#include <stdio.h>
#include <stdlib.h>
#include "matmult.h"
#include "cblas.h"

// allocate a double-prec m x n matrix
double **
dmalloc_2d(int m, int n)
{
    if (m <= 0 || n <= 0)
        return NULL;
    double **A = malloc(m * sizeof(double *));
    if (A == NULL)
        return NULL;
    A[0] = malloc(m * n * sizeof(double));
    if (A[0] == NULL)
    {
        free(A);
        return NULL;
    }
    int i;
    for (i = 1; i < m; i++)
        A[i] = A[0] + i * n;
    return A;
}

void dfree_2d(double **A)
{
    free(A[0]);
    free(A);
}

// https://github.com/attractivechaos/matmul/blob/master/matmul.c
double **mat_transpose(int n_rows, int n_cols, double *const *a)
{
    int i, j;
    double **m;
    m = dmalloc_2d(n_cols, n_rows);
    if (m == NULL)
    {
        printf("Couldn't allocate new matrix!\n");
        exit(1);
    }
    for (i = 0; i < n_rows; ++i)
        for (j = 0; j < n_cols; ++j)
            m[j][i] = a[i][j];
    return m;
}

void matmult_nat(int m, int n, int k, double **A, double **B, double **C)
{
    int i, j, l;
    double **BT = mat_transpose(k, n, B);
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            C[i][j] = 0;
            for (l = 0; l < k; l++)
            {
                C[i][j] += A[i][l] * BT[j][l];
            }
        }
    }
    dfree_2d(BT);
}

// void matmult_nat(int m,int n,int k,double **A,double **B,double **C)
// {
//     int i,j,l;
//     for (i = 0; i < m; i++)
//     {
//         for (j = 0; j < n; j++)
//         {
//             C[i][j] = 0;
//             for (l = 0; l < k; l++)
//             {
//                 C[i][j] += A[i][l] * B[l][j];
//             }
//         }
//     }
// }

void matmult_lib(int m, int n, int k, double **A, double **B, double **C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A[0], k, B[0], n, 0, C[0], n);
}

void matmult_mkn(int m, int n, int k, double **A, double **B, double **C)
{
}

void matmult_mnk(int m, int n, int k, double **A, double **B, double **C)
{
}

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C)
{
}

void matmult_knm(int m, int n, int k, double **A, double **B, double **C)
{
}

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C)
{
}

void matmult_nkm(int m, int n, int k, double **A, double **B, double **C)
{
}