// todo: parallelize the sequential code
#include "mpi.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>

double *init_A(int n);
double *init_v(int n);
double *init_w(int n);
double row_Mat_col(double *w, double *A, double *v, int num_rows,
                   int num_columns);
double svv(double s, double *A, double *w, int num_elems);

int main(int argc, char **argv) {
    int n = 10000;

    if (argc == 2)
        n = atoi(argv[1]);

    int num_procs;
    int rank;

#ifdef NOMPI
    num_procs = 1;
    rank = 0;
#else
    // MPI SET UP
    int provided;
    int required = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, required, &provided);

    // MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    int rank_rows[num_procs];
    int chunk_counts[num_procs];
    int rows_per_rank = n / num_procs; // whole division
    int rem = n - num_procs * rows_per_rank;

    // each rank will recieve n/num_procs rows and the remaining will be shared
    for (int idx = 0; idx < num_procs; idx++) {
        rank_rows[idx] = rows_per_rank + (rem > idx);
        chunk_counts[idx] = rank_rows[idx] * n;
    }

    int A_disps[num_procs], v_disps[num_procs];
    A_disps[0] = 0;
    v_disps[0] = 0;

    for (int i = 1; i < num_procs; i++) {
        A_disps[i] = A_disps[i - 1] + chunk_counts[i - 1];
        v_disps[i] = v_disps[i - 1] + rank_rows[i - 1];
    }

    // allocate space for each array for each proccess
    double *A_p = malloc(chunk_counts[rank] * sizeof(double));
    double *v_p = malloc(rank_rows[rank] * sizeof(double));
    double *w_p = malloc(n * sizeof(double));

    double *A_all = NULL;
    double *v_all = NULL;

    double final_res;
    if (rank == 0) {
        A_all = init_A(n);
        v_all = init_v(n);
        w_p = init_w(n);
    }

    // timeit after initialization;
    double t_start = omp_get_wtime();

#ifdef NOMPI
    A_p = A_all;
    v_p = v_all;
#else
    // send each process its relative data
    MPI_Request request0, request1, request2;
    MPI_Status sta0, sta1, sta2;
    int tag0 = 68, tag1 = 419, tag2 = 13;

    MPI_Iscatterv(A_all, chunk_counts, A_disps, MPI_DOUBLE, A_p,
                  chunk_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD, &request0);
    MPI_Iscatterv(v_all, rank_rows, v_disps, MPI_DOUBLE, v_p,
                  chunk_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD, &request1);
    MPI_Ibcast(w_p, n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request2);

    MPI_Wait(&request0, &sta0);
    MPI_Wait(&request1, &sta1);
    MPI_Wait(&request2, &sta2);
#endif

    double t_data = omp_get_wtime();

    // calculate subproblems
    double res = 0.;
    res = row_Mat_col(v_p, A_p, w_p, rank_rows[rank], n);

#ifdef NOMPI
    final_res = res;
#else
    MPI_Reduce(&res, &final_res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

    // timeit
    double t_end = omp_get_wtime();

    if (rank == 0) {
        printf("took %f seconds total, data transmition %f seconds \n",
               t_end - t_start, t_data - t_start);
        printf("Result: %f \n", final_res);
        free(A_all);
        free(v_all);
    }

#ifndef NOMPI
    /// free memory
    free(A_p);
    free(v_p);
    free(w_p);

    MPI_Finalize();
#endif

    return 0;
}

double row_Mat_col(double *v, double *A, double *w, int num_rows,
                   int num_columns) {

    double t_start = omp_get_wtime();

    double res = 0.;
    double vMv;

#pragma omp parallel for reduction(+ : res) private(vMv)
    for (int row = 0; row < num_rows; row++) {
        vMv = svv(*(v + row), A + row * num_columns, w, num_columns);
        res += vMv;
    }

    double t_end = omp_get_wtime();
    printf("Calculation time: %f \n", t_end - t_start);

    return res;
}

double svv(double s, double *A, double *w, int num_elems) {
    double vv = 0.;

#pragma omp parallel for simd reduction(+ : vv)
    for (int idx = 0; idx < num_elems; idx++) {
        vv += A[idx] * w[idx];
    }

    return s * vv;
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
