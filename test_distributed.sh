#!/bin/bash
# test_distributed.sh - Automated testing for distributed implementations

echo "Compiling distributed matrix multiplication programs..."

# Compile MPI version
mpicc -o mpi_matrix mpi_matrix.c -lm
if [ $? -ne 0 ]; then
    echo "Error compiling MPI version"
    exit 1
fi

# Compile MPI+OpenMP version
mpicc -o mpi_openmp_matrix mpi_openmp_matrix.c -fopenmp -lm
if [ $? -ne 0 ]; then
    echo "Error compiling MPI+OpenMP version"
    exit 1
fi

# Compile MPI+OpenCL version
mpicc -o mpi_opencl_matrix mpi_opencl_matrix.c -lOpenCL -lm
if [ $? -ne 0 ]; then
    echo "Warning: MPI+OpenCL compilation failed (OpenCL may not be available)"
fi

echo "Compilation completed!"

SIZES=(500 1000 1500)
PROCESSES=(1 2 4)
THREADS=(2 4 8)

echo "Matrix Size,Processes,Threads,MPI(s),MPI+OpenMP(s),MPI+OpenCL(s)" > distributed_results.csv

for size in "${SIZES[@]}"; do
    echo "Testing matrix size: $size"
    
    for proc in "${PROCESSES[@]}"; do
        echo "  Testing with $proc processes..."
        
        # Run MPI version
        mpi_output=$(mpirun -np $proc ./mpi_matrix $size 2>/dev/null)
        mpi_time=$(echo "$mpi_output" | grep "Execution Time" | awk '{print $3}')
        
        for thread in "${THREADS[@]}"; do
            echo "    Testing with $thread threads per process..."
            
            # Run MPI+OpenMP version
            openmp_output=$(mpirun -np $proc ./mpi_openmp_matrix $size $thread 2>/dev/null)
            openmp_time=$(echo "$openmp_output" | grep "Execution Time" | awk '{print $3}')
            
            # Run MPI+OpenCL version (if available)
            opencl_time="N/A"
            if [ -f "./mpi_opencl_matrix" ]; then
                opencl_output=$(mpirun -np $proc ./mpi_opencl_matrix $size 2>/dev/null)
                opencl_time=$(echo "$opencl_output" | grep "Execution Time" | awk '{print $3}')
            fi
            
            echo "$size,$proc,$thread,$mpi_time,$openmp_time,$opencl_time" >> distributed_results.csv
        done
    done
done

echo "Testing complete! Results saved to distributed_results.csv"
echo ""
echo "Results Summary:"
cat distributed_results.csv