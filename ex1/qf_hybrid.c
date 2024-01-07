// todo: parallelize the sequential code
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double *init_A(int n);
double *init_v(int n);
double *init_w(int n);

int main(int argc, char **argv) {
    int n = 8;

    if (argc == 2)
        n = atoi(argv[1]);

    int num_procs;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rank_rows[num_procs];
    int chunk_counts[num_procs];
    int rows_per_rank = n / num_procs; // whole division
    int rem = n - num_procs * rows_per_rank;

    // each rank will recieve n/num_procs rows and the remaining will be shared
    for (int idx = 0; idx < num_procs; idx++) {
        rank_rows[idx] = rows_per_rank + (rem > idx);
        chunk_counts[idx] = rank_rows[idx] * n;
    }

    int disps[num_procs];
    disps[0] = 0;

    for (int i = 1; i < num_procs; i++) {
        disps[i] = disps[i - 1] + chunk_counts[i - 1];
    }

    // allocate space for each array for each proccess
    double *A_p = malloc(chunk_counts[rank] * sizeof(double));
    double *v_p = malloc(sizeof(double));
    double *w_p = malloc(n * sizeof(double));

    double *A_all = NULL;
    double *v_all = NULL;
    double *w_all = NULL;

    if (rank == 0) {
        A_all = init_A(n);
        v_all = init_v(n);
        w_all = init_w(n);

        for(int i = 0; i<num_procs; i++){
            printf("%d %d %d %d\n", i, chunk_counts[i], disps[i], rank_rows[i]);
        }
        printf("\n");

        for (int i = 0; i < n; i++) {
            printf("%d \t\t", i);
            for (int j = 0; j < n; j++) {
                printf("%3.2f \t", A_all[i*n + j]);
            }
            printf("\n");
        }
    }

    MPI_Scatterv(A_all, chunk_counts, disps, MPI_DOUBLE, A_p,
                 chunk_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(A_all);
        free(v_all);
        free(w_all);
    }

    int count = 0;
    for (int i = 0; A_p[i] != '\0'; i++) {
        count++;
    }

    if (2 == rank) {
        printf("\n");

        for (int i = 0; i < rank_rows[rank]; i++) {
            for (int j = 0; j < n; j++) {
                printf("%3.2f \t", A_p[i * n + j]);
            }
            printf("\n");
        }
    }

    /// free memory
    free(A_p);
    free(v_p);
    free(w_p);

    // timeit
    // double t_start = omp_get_wtime();

    /// compute
    double result = 0.;
    /*
        for (int i=0; i<n; ++i)
                for (int j=0; j<n; ++j)
                        result += v_all[i] * A_all[i*n + j] * w_all[j];
    */

    MPI_Finalize();

    // timeit
    // double t_end = omp_get_wtime();
    // printf("took %f seconds", t_end - t_start);

    printf("Result = %lf\n", result);

    return 0;
}

double *init_A(int n) {
    double *A = (double *)malloc(n * n * sizeof(double));

    /// init A_ij = (i + 2*j) / n^2
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[i * n + j] = (i + 2.0 * j) / (n * n);

    return A;
}

double *init_v(int n) {
    double *v = (double *)malloc(n * sizeof(double));

    /// init v_i = 1 + 2 / (i+0.5)
    for (int i = 0; i < n; ++i)
        v[i] = 1.0 + 2.0 / (i + 0.5);

    return v;
}

double *init_w(int n) {
    double *w = (double *)malloc(n * sizeof(double));

    /// init w_i = 1 - i / (3.*n)
    for (int i = 0; i < n; ++i)
        w[i] = 1.0 - i / (3.0 * n);

    return w;
}
