#!/usr/bin/env python3
import numpy as np
import time
import sys
from multiprocessing import Pool, cpu_count

def matrix_multiply_sequential(A, B):
    """Sequential matrix multiplication"""
    return np.dot(A, B)

def matrix_multiply_parallel_chunk(args):
    """Parallel matrix multiplication worker function"""
    A_chunk, B, start_row = args
    return np.dot(A_chunk, B)

def matrix_multiply_parallel(A, B, num_processes=None):
    """Parallel matrix multiplication using multiprocessing"""
    if num_processes is None:
        num_processes = cpu_count()
    
    n = A.shape[0]
    chunk_size = n // num_processes
    
    # Split matrix A into chunks
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else n
        chunks.append((A[start:end], B, start))
    
    # Process chunks in parallel
    with Pool(num_processes) as pool:
        results = pool.map(matrix_multiply_parallel_chunk, chunks)
    
    # Combine results
    return np.vstack(results)

def run_matrix_test(n=1000, num_processes=4):
    """Run matrix multiplication test"""
    print(f"Python Matrix Multiplication Test")
    print(f"Matrix Size: {n}x{n}")
    print(f"Number of Processes: {num_processes}")
    print("-" * 40)
    
    # Initialize matrices
    np.random.seed(42)
    A = np.random.rand(n, n) * 10.0
    B = np.random.rand(n, n) * 10.0
    
    # Sequential multiplication
    print("Running sequential multiplication...")
    start_time = time.time()
    C_seq = matrix_multiply_sequential(A, B)
    seq_time = time.time() - start_time
    print(f"Sequential Time: {seq_time:.4f} seconds")
    
    # Parallel multiplication
    print("Running parallel multiplication...")
    start_time = time.time()
    C_par = matrix_multiply_parallel(A, B, num_processes)
    par_time = time.time() - start_time
    print(f"Parallel Time: {par_time:.4f} seconds")
    
    # Calculate speedup
    speedup = seq_time / par_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results are similar
    if np.allclose(C_seq, C_par, rtol=1e-10):
        print("Results verified - parallel matches sequential")
    else:
        print("Results differ - verification failed")
    
    # Save results
    print("\nSaving results...")
    np.savetxt("output_python_sequential.txt", C_seq, fmt="%.2f")
    np.savetxt("output_python_parallel.txt", C_par, fmt="%.2f")
    
    return {
        'sequential_time': seq_time,
        'parallel_time': par_time,
        'speedup': speedup,
        'matrix_size': n,
        'processes': num_processes
    }

if __name__ == "__main__":
    # Parse command line arguments
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else cpu_count()
    
    # Run test
    results = run_matrix_test(n, processes)
    
    print(f"\nTest completed successfully!")
    print(f"Check output files: output_python_sequential.txt, output_python_parallel.txt")