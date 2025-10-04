#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void initialize_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (double)rand() / RAND_MAX * 10.0;
    }
}

void matrix_multiply_block(double *A, double *B, double *C, int n, int start_row, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size, n = 1000;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) n = atoi(argv[1]);
    
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    
    int rows_per_proc = n / size;
    int remainder = n % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate local matrices
    local_A = (double*)malloc(local_rows * n * sizeof(double));
    local_C = (double*)malloc(local_rows * n * sizeof(double));
    
    if (rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        C = (double*)malloc(n * n * sizeof(double));
        
        srand(time(NULL));
        initialize_matrix(A, n, n);
        initialize_matrix(B, n, n);
        
        printf("MPI Matrix Multiplication\n");
        printf("Matrix Size: %dx%d\n", n, n);
        printf("Number of Processes: %d\n", size);
    }
    
    // Allocate B on all processes
    if (rank != 0) {
        B = (double*)malloc(n * n * sizeof(double));
    }
    
    // Broadcast matrix B to all processes
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Scatter matrix A rows
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    start_time = MPI_Wtime();
    
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, 
                 local_A, local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform local matrix multiplication
    matrix_multiply_block(local_A, B, local_C, n, 0, local_rows);
    
    // Gather results
    MPI_Gatherv(local_C, local_rows * n, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Execution Time: %.4f seconds\n", end_time - start_time);
        
        // Write result to file
        FILE *file = fopen("output_mpi.txt", "w");
        if (file) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    fprintf(file, "%.2f ", C[i * n + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
        
        free(A);
        free(C);
    }
    
    free(local_A);
    free(local_C);
    free(B);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}