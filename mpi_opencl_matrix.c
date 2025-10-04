#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>
#include <time.h>

const char* kernel_source = 
"__kernel void matrix_multiply(__global float* A, __global float* B, __global float* C, int n, int local_rows) {\n"
"    int i = get_global_id(0);\n"
"    int j = get_global_id(1);\n"
"    \n"
"    if (i < local_rows && j < n) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < n; k++) {\n"
"            sum += A[i * n + k] * B[k * n + j];\n"
"        }\n"
"        C[i * n + j] = sum;\n"
"    }\n"
"}\n";

void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 10.0f;
    }
}

void check_error(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        printf("Error during %s: %d\n", operation, error);
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    int rank, size, n = 1000;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) n = atoi(argv[1]);
    
    float *A = NULL, *B = NULL, *C = NULL;
    float *local_A, *local_C;
    
    int rows_per_proc = n / size;
    int remainder = n % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate local matrices
    local_A = (float*)malloc(local_rows * n * sizeof(float));
    local_C = (float*)malloc(local_rows * n * sizeof(float));
    
    if (rank == 0) {
        A = (float*)malloc(n * n * sizeof(float));
        B = (float*)malloc(n * n * sizeof(float));
        C = (float*)malloc(n * n * sizeof(float));
        
        srand(time(NULL));
        initialize_matrix(A, n, n);
        initialize_matrix(B, n, n);
        
        printf("MPI + OpenCL Hybrid Matrix Multiplication\n");
        printf("Matrix Size: %dx%d\n", n, n);
        printf("Number of MPI Processes: %d\n", size);
    }
    
    // Allocate B on all processes
    if (rank != 0) {
        B = (float*)malloc(n * n * sizeof(float));
    }
    
    // Broadcast matrix B to all processes
    MPI_Bcast(B, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Scatter matrix A rows
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT, 
                 local_A, local_rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int error;
    
    // Get platform and device
    error = clGetPlatformIDs(1, &platform, NULL);
    check_error(error, "getting platform");
    
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (error != CL_SUCCESS) {
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    check_error(error, "getting device");
    
    // Create context and queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    check_error(error, "creating context");
    
    queue = clCreateCommandQueue(context, device, 0, &error);
    check_error(error, "creating command queue");
    
    // Create and build program
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error);
    check_error(error, "creating program");
    
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    check_error(error, "building program");
    
    // Create kernel
    kernel = clCreateKernel(program, "matrix_multiply", &error);
    check_error(error, "creating kernel");
    
    start_time = MPI_Wtime();
    
    // Create buffers
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     local_rows * n * sizeof(float), local_A, &error);
    check_error(error, "creating buffer A");
    
    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     n * n * sizeof(float), B, &error);
    check_error(error, "creating buffer B");
    
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     local_rows * n * sizeof(float), NULL, &error);
    check_error(error, "creating buffer C");
    
    // Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
    error |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    error |= clSetKernelArg(kernel, 4, sizeof(int), &local_rows);
    check_error(error, "setting kernel arguments");
    
    // Execute kernel
    size_t global_work_size[2] = {local_rows, n};
    error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_error(error, "executing kernel");
    
    // Read result
    error = clEnqueueReadBuffer(queue, buffer_C, CL_TRUE, 0, local_rows * n * sizeof(float), local_C, 0, NULL, NULL);
    check_error(error, "reading result");
    
    // Gather results
    MPI_Gatherv(local_C, local_rows * n, MPI_FLOAT,
                C, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("Execution Time: %.4f seconds\n", end_time - start_time);
        
        // Write result to file
        FILE *file = fopen("output_mpi_opencl.txt", "w");
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
    
    // Cleanup OpenCL
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(local_A);
    free(local_C);
    free(B);
    free(sendcounts);
    free(displs);
    
    MPI_Finalize();
    return 0;
}