/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "alloc3d.h"
#include "print.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int          N = N_DEFAULT;
    int          iter_max = 1000;
    double         tolerance;
    double         start_T;
    int                  output_type = 0;
    char         *output_prefix = "poisson_res";
    char        *output_ext    = "";
    char         output_filename[FILENAME_MAX];
    double          ***u = NULL;

    if (argc < 5) {
        printf("More arguments please\n");
        return 0;
    }

    /* get the paramters from the command line */
    N         = atoi(argv[1]);         // grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
         output_type = atoi(argv[5]);  // ouput type
    }

    // allocate memory
    if ( (u = d_malloc_3d(N+2, N+2, N+2)) == NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    // i = x
    // j = y
    // k = z
    for (int i = 0; i < N+2; ++i) {
      for (int j = 0; j < N+2; ++j) {
        for (int k = 0; k < N+2; ++k) {
#if CHECK_CORRECTNESS
          u[i][j][k] = 0;
#else
          u[i][j][k] = start_T;
#endif
        }
      }
    }
    for (int i = 0; i < N+2; ++i) {
      for (int j = 0; j < N+2; ++j) {
#if CHECK_CORRECTNESS
        u[i][j][0] = 0.0;
        u[i][j][N+1] = 0.0;
        u[i][0][j] = 0.0;
        u[i][N+1][j] = 0.0;
        u[0][i][j] = 0.0;
        u[N+1][i][j] = 0.0;
#else
        u[i][j][0] = 20.0;
        u[i][j][N+1] = 20.0;
        u[i][0][j] = 0.0;
        u[i][N+1][j] = 20.0;
        u[0][i][j] = 20.0;
        u[N+1][i][j] = 20.0;
#endif
      }
    }

    #ifdef _JACOBI
    u = jacobi(N, iter_max, tolerance, u);
    #endif
    #ifdef _GAUSS_SEIDEL_H
    gauss_seidel(N, iter_max, tolerance, u);
    #endif

#if CHECK_CORRECTNESS
    double delta = 2.0 / (double) N;
    for (int i = 1; i < N+1; ++i) {
      double x = -1.0 + (i * delta);
      for (int j = 1; j < N+1; ++j) {
        double y = -1.0 + (j * delta);
        for (int k = 1; k < N+1; ++k) {
            double z = -1.0 + (k * delta);
            double actual = sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
            if (fabs(u[i][j][k] - actual) > .1) {
              printf("Solution is not correct! At the index: [%d][%d][%d]\n" 
                     "Diff from true: %f - %f = %f.\n",i,j,k, 
                     u[i][j][k], actual, fabs(u[i][j][k] - actual));
            }
        }
      }
    }
#endif

    // dump  results if wanted 
    switch(output_type) {
         case 0:
             // no output at all
             break;
         case 3:
             output_ext = ".bin";
             sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
             fprintf(stderr, "Write binary dump to %s: ", output_filename);
             print_binary(output_filename, N, u);
             break;
         case 4:
             output_ext = ".vtk";
             sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
             fprintf(stderr, "Write VTK file to %s: ", output_filename);
             print_vtk(output_filename, N, u);
             break;
         default:
             fprintf(stderr, "Non-supported output type!\n");
             break;
    }

    // de-allocate memory
    free(u);

    return(0);
}
