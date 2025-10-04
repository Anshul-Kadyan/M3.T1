#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_distributed_performance():
    # Read results
    if not os.path.exists('distributed_results.csv'):
        print("Error: distributed_results.csv not found. Please run test_distributed.sh first.")
        return
    
    df = pd.read_csv('distributed_results.csv')
    
    # Convert time columns to numeric, handling 'N/A' values
    time_cols = ['MPI(s)', 'MPI+OpenMP(s)', 'MPI+OpenCL(s)']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate speedup relative to single process MPI
    baseline = df[(df['Processes'] == 1) & (df['Threads'] == 2)].groupby('Matrix Size')['MPI(s)'].first()
    
    for size in df['Matrix Size'].unique():
        size_mask = df['Matrix Size'] == size
        if size in baseline.index:
            base_time = baseline[size]
            df.loc[size_mask, 'MPI_Speedup'] = base_time / df.loc[size_mask, 'MPI(s)']
            df.loc[size_mask, 'OpenMP_Speedup'] = base_time / df.loc[size_mask, 'MPI+OpenMP(s)']
            if not df.loc[size_mask, 'MPI+OpenCL(s)'].isna().all():
                df.loc[size_mask, 'OpenCL_Speedup'] = base_time / df.loc[size_mask, 'MPI+OpenCL(s)']
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Execution Time vs Processes
    sizes = df['Matrix Size'].unique()
    processes = df['Processes'].unique()
    
    for size in sizes:
        size_data = df[df['Matrix Size'] == size].groupby('Processes').mean()
        ax1.plot(size_data.index, size_data['MPI(s)'], 'o-', label=f'MPI (N={size})')
        ax1.plot(size_data.index, size_data['MPI+OpenMP(s)'], 's-', label=f'MPI+OpenMP (N={size})')
        if not size_data['MPI+OpenCL(s)'].isna().all():
            ax1.plot(size_data.index, size_data['MPI+OpenCL(s)'], '^-', label=f'MPI+OpenCL (N={size})')
    
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time vs Number of Processes')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Speedup vs Processes
    for size in sizes:
        size_data = df[df['Matrix Size'] == size].groupby('Processes').mean()
        if 'MPI_Speedup' in size_data.columns:
            ax2.plot(size_data.index, size_data['MPI_Speedup'], 'o-', label=f'MPI (N={size})')
        if 'OpenMP_Speedup' in size_data.columns:
            ax2.plot(size_data.index, size_data['OpenMP_Speedup'], 's-', label=f'MPI+OpenMP (N={size})')
        if 'OpenCL_Speedup' in size_data.columns and not size_data['OpenCL_Speedup'].isna().all():
            ax2.plot(size_data.index, size_data['OpenCL_Speedup'], '^-', label=f'MPI+OpenCL (N={size})')
    
    # Ideal speedup line
    max_processes = df['Processes'].max()
    ax2.plot(processes, processes, 'k--', label='Ideal Speedup')
    
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Number of Processes')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Performance vs Threads (for hybrid implementations)
    threads = df['Threads'].unique()
    avg_by_threads = df.groupby('Threads').mean()
    
    ax3.bar(threads - 0.2, avg_by_threads['MPI(s)'], 0.2, label='MPI', alpha=0.7)
    ax3.bar(threads, avg_by_threads['MPI+OpenMP(s)'], 0.2, label='MPI+OpenMP', alpha=0.7)
    if not avg_by_threads['MPI+OpenCL(s)'].isna().all():
        ax3.bar(threads + 0.2, avg_by_threads['MPI+OpenCL(s)'], 0.2, label='MPI+OpenCL', alpha=0.7)
    
    ax3.set_xlabel('Threads per Process')
    ax3.set_ylabel('Average Execution Time (seconds)')
    ax3.set_title('Performance vs Threads per Process')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Scalability Analysis
    for impl in ['MPI', 'MPI+OpenMP', 'MPI+OpenCL']:
        col_name = f'{impl}(s)'
        if col_name in df.columns and not df[col_name].isna().all():
            efficiency_data = []
            process_counts = []
            
            for proc in processes:
                proc_data = df[df['Processes'] == proc][col_name].mean()
                baseline_data = df[df['Processes'] == 1][col_name].mean()
                
                if not pd.isna(proc_data) and not pd.isna(baseline_data):
                    speedup = baseline_data / proc_data
                    efficiency = (speedup / proc) * 100
                    efficiency_data.append(efficiency)
                    process_counts.append(proc)
            
            if efficiency_data:
                ax4.plot(process_counts, efficiency_data, 'o-', label=impl)
    
    ax4.axhline(y=100, color='k', linestyle='--', label='100% Efficiency')
    ax4.set_xlabel('Number of Processes')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('Parallel Efficiency')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('distributed_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate comprehensive report
    with open('distributed_performance_report.txt', 'w') as f:
        f.write("DISTRIBUTED MATRIX MULTIPLICATION PERFORMANCE ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("IMPLEMENTATIONS TESTED:\n")
        f.write("-" * 22 + "\n")
        f.write("1. MPI - Pure message passing\n")
        f.write("2. MPI + OpenMP - Hybrid distributed + shared memory\n")
        f.write("3. MPI + OpenCL - Hybrid distributed + GPU acceleration\n\n")
        
        f.write("TEST CONFIGURATION:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Matrix sizes: {list(sizes)}\n")
        f.write(f"Process counts: {list(processes)}\n")
        f.write(f"Thread counts: {list(threads)}\n")
        f.write(f"Total test configurations: {len(df)}\n\n")
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 19 + "\n")
        
        # Best performance for each implementation
        for impl in ['MPI(s)', 'MPI+OpenMP(s)', 'MPI+OpenCL(s)']:
            if impl in df.columns and not df[impl].isna().all():
                best_idx = df[impl].idxmin()
                best_row = df.loc[best_idx]
                f.write(f"Best {impl.replace('(s)', '')}: {best_row[impl]:.4f}s ")
                f.write(f"(N={best_row['Matrix Size']}, P={best_row['Processes']}, T={best_row['Threads']})\n")
        
        f.write("\n")
        
        # Average performance by process count
        f.write("AVERAGE PERFORMANCE BY PROCESS COUNT:\n")
        f.write("-" * 37 + "\n")
        avg_by_proc = df.groupby('Processes')[time_cols].mean().round(4)
        f.write(avg_by_proc.to_string())
        f.write("\n\n")
        
        # Scalability analysis
        f.write("SCALABILITY ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        
        baseline_1proc = df[df['Processes'] == 1].groupby('Matrix Size')[time_cols].mean()
        max_proc_data = df[df['Processes'] == df['Processes'].max()].groupby('Matrix Size')[time_cols].mean()
        
        for size in sizes:
            f.write(f"\nMatrix Size {size}x{size}:\n")
            for impl in time_cols:
                if impl in baseline_1proc.columns and size in baseline_1proc.index:
                    if not pd.isna(baseline_1proc.loc[size, impl]) and not pd.isna(max_proc_data.loc[size, impl]):
                        speedup = baseline_1proc.loc[size, impl] / max_proc_data.loc[size, impl]
                        f.write(f"  {impl.replace('(s)', '')} max speedup: {speedup:.2f}x\n")
        
        f.write("\n\nDETAILED RESULTS:\n")
        f.write("-" * 16 + "\n")
        f.write(df.to_string(index=False))
    
    print("Distributed analysis complete!")
    print("Generated files:")
    print("- distributed_performance_analysis.png: Visual performance charts")
    print("- distributed_performance_report.txt: Detailed analysis report")

if __name__ == "__main__":
    analyze_distributed_performance()