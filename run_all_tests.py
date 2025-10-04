#!/usr/bin/env python3
"""
Complete test runner for distributed matrix multiplication project
Since MPI is not available, this demonstrates the concepts using Python multiprocessing
"""
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os

def matrix_multiply_chunk(args):
    """Worker function for parallel matrix multiplication"""
    A_chunk, B, chunk_id = args
    return chunk_id, np.dot(A_chunk, B)

def run_matrix_test(n, num_processes, test_name):
    """Run a single matrix multiplication test"""
    print(f"\n{test_name}")
    print(f"Matrix Size: {n}x{n}, Processes: {num_processes}")
    print("-" * 50)
    
    # Initialize matrices
    np.random.seed(42)
    A = np.random.rand(n, n) * 10.0
    B = np.random.rand(n, n) * 10.0
    
    # Sequential baseline
    start_time = time.time()
    C_sequential = np.dot(A, B)
    sequential_time = time.time() - start_time
    
    # Parallel version
    start_time = time.time()
    
    if num_processes == 1:
        C_parallel = np.dot(A, B)
    else:
        # Split matrix A into chunks
        chunk_size = n // num_processes
        chunks = []
        for i in range(num_processes):
            start_row = i * chunk_size
            end_row = start_row + chunk_size if i < num_processes - 1 else n
            chunks.append((A[start_row:end_row], B, i))
        
        # Process in parallel
        with Pool(num_processes) as pool:
            results = pool.map(matrix_multiply_chunk, chunks)
        
        # Combine results
        results.sort(key=lambda x: x[0])  # Sort by chunk_id
        C_parallel = np.vstack([result[1] for result in results])
    
    parallel_time = time.time() - start_time
    
    # Calculate metrics
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    efficiency = speedup / num_processes * 100
    
    print(f"Sequential Time: {sequential_time:.4f}s")
    print(f"Parallel Time: {parallel_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")
    
    # Verify correctness
    if np.allclose(C_sequential, C_parallel, rtol=1e-10):
        print("Results verified: PASS")
    else:
        print("Results verification: FAIL")
    
    return {
        'test_name': test_name,
        'matrix_size': n,
        'processes': num_processes,
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'efficiency': efficiency
    }

def main():
    """Run comprehensive tests simulating the distributed matrix project"""
    print("=" * 60)
    print("DISTRIBUTED MATRIX MULTIPLICATION PROJECT")
    print("Python Implementation (MPI simulation)")
    print("=" * 60)
    
    # Test configurations
    matrix_sizes = [500, 1000, 1500]
    process_counts = [1, 2, 4, 8]
    
    results = []
    
    # Run all test combinations
    for n in matrix_sizes:
        for p in process_counts:
            if p <= cpu_count():  # Don't exceed available cores
                test_name = f"Matrix {n}x{n} with {p} processes"
                result = run_matrix_test(n, p, test_name)
                results.append(result)
    
    # Save results to CSV
    csv_file = 'distributed_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\nResults saved to {csv_file}")
    
    # Generate performance analysis
    generate_performance_plots(results)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("\nProject execution completed!")
    print("Generated files:")
    print("- distributed_results.csv")
    print("- performance_analysis.png")
    print("- performance_report.txt")

def generate_performance_plots(results):
    """Generate performance visualization"""
    # Group results by matrix size
    sizes = sorted(set(r['matrix_size'] for r in results))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distributed Matrix Multiplication Performance Analysis', fontsize=16)
    
    # Speedup vs Processes
    for size in sizes:
        size_results = [r for r in results if r['matrix_size'] == size]
        processes = [r['processes'] for r in size_results]
        speedups = [r['speedup'] for r in size_results]
        ax1.plot(processes, speedups, 'o-', label=f'{size}x{size}')
    
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Number of Processes')
    ax1.legend()
    ax1.grid(True)
    
    # Efficiency vs Processes
    for size in sizes:
        size_results = [r for r in results if r['matrix_size'] == size]
        processes = [r['processes'] for r in size_results]
        efficiencies = [r['efficiency'] for r in size_results]
        ax2.plot(processes, efficiencies, 's-', label=f'{size}x{size}')
    
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency vs Number of Processes')
    ax2.legend()
    ax2.grid(True)
    
    # Execution Time vs Matrix Size
    processes_list = sorted(set(r['processes'] for r in results))
    for p in processes_list:
        proc_results = [r for r in results if r['processes'] == p]
        sizes_p = [r['matrix_size'] for r in proc_results]
        times = [r['parallel_time'] for r in proc_results]
        ax3.plot(sizes_p, times, '^-', label=f'{p} processes')
    
    ax3.set_xlabel('Matrix Size')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title('Execution Time vs Matrix Size')
    ax3.legend()
    ax3.grid(True)
    
    # Speedup Heatmap
    matrix_data = np.zeros((len(sizes), len(processes_list)))
    for i, size in enumerate(sizes):
        for j, proc in enumerate(processes_list):
            matching = [r for r in results if r['matrix_size'] == size and r['processes'] == proc]
            if matching:
                matrix_data[i, j] = matching[0]['speedup']
    
    im = ax4.imshow(matrix_data, cmap='viridis', aspect='auto')
    ax4.set_xticks(range(len(processes_list)))
    ax4.set_xticklabels(processes_list)
    ax4.set_yticks(range(len(sizes)))
    ax4.set_yticklabels(sizes)
    ax4.set_xlabel('Number of Processes')
    ax4.set_ylabel('Matrix Size')
    ax4.set_title('Speedup Heatmap')
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, label='Speedup')
    
    # Add text annotations
    for i in range(len(sizes)):
        for j in range(len(processes_list)):
            text = ax4.text(j, i, f'{matrix_data[i, j]:.1f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results):
    """Generate detailed performance report"""
    with open('performance_report.txt', 'w') as f:
        f.write("DISTRIBUTED MATRIX MULTIPLICATION PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Test Configuration:\n")
        f.write(f"- CPU Cores Available: {cpu_count()}\n")
        f.write(f"- Matrix Sizes Tested: {sorted(set(r['matrix_size'] for r in results))}\n")
        f.write(f"- Process Counts: {sorted(set(r['processes'] for r in results))}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 40 + "\n")
        
        for result in results:
            f.write(f"\nTest: {result['test_name']}\n")
            f.write(f"  Sequential Time: {result['sequential_time']:.4f}s\n")
            f.write(f"  Parallel Time: {result['parallel_time']:.4f}s\n")
            f.write(f"  Speedup: {result['speedup']:.2f}x\n")
            f.write(f"  Efficiency: {result['efficiency']:.1f}%\n")
        
        # Best performance analysis
        f.write("\n" + "=" * 40 + "\n")
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("=" * 40 + "\n")
        
        best_speedup = max(results, key=lambda x: x['speedup'])
        f.write(f"\nBest Speedup: {best_speedup['speedup']:.2f}x\n")
        f.write(f"Configuration: {best_speedup['matrix_size']}x{best_speedup['matrix_size']} matrix, {best_speedup['processes']} processes\n")
        
        best_efficiency = max(results, key=lambda x: x['efficiency'])
        f.write(f"\nBest Efficiency: {best_efficiency['efficiency']:.1f}%\n")
        f.write(f"Configuration: {best_efficiency['matrix_size']}x{best_efficiency['matrix_size']} matrix, {best_efficiency['processes']} processes\n")
        
        f.write("\nConclusions:\n")
        f.write("- Python's numpy uses optimized BLAS libraries, making sequential very fast\n")
        f.write("- Multiprocessing overhead dominates for small/medium matrices\n")
        f.write("- Real MPI implementation would show better scaling characteristics\n")
        f.write("- GPU acceleration (OpenCL) would provide significant speedup for large matrices\n")

if __name__ == "__main__":
    main()