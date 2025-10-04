# SIT315 M3.T1P: Distributed Matrix Multiplication

This project implements distributed and hybrid parallel matrix multiplication using:
1. **MPI** - Pure message passing interface
2. **MPI + OpenMP** - Hybrid distributed + shared memory
3. **MPI + OpenCL** - Hybrid distributed + GPU acceleration

## Project Structure

```
module3_matrix/
├── mpi_matrix.c                 # Pure MPI implementation
├── mpi_openmp_matrix.c          # MPI + OpenMP hybrid
├── mpi_opencl_matrix.c          # MPI + OpenCL hybrid
├── test_distributed.sh          # Linux/Mac testing script
├── compile_distributed.bat      # Windows compilation script
├── analyze_distributed.py       # Performance analysis
└── README_Module3.md            # This file
```

## Prerequisites

### Required Software:
- **MPI Implementation**: 
  - Windows: Microsoft MPI or MPICH
  - Linux: OpenMPI or MPICH
- **C Compiler**: GCC or compatible
- **OpenMP**: Usually included with GCC
- **OpenCL** (optional): GPU vendor SDK

### Installation:

#### Windows:
1. Install Microsoft MPI: https://www.microsoft.com/en-us/download/details.aspx?id=57467
2. Install MinGW-w64 or Visual Studio Build Tools
3. For OpenCL: Install GPU vendor SDK (NVIDIA CUDA, AMD APP, Intel OpenCL)

#### Linux:
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin openmpi-dev
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# CentOS/RHEL
sudo yum install openmpi openmpi-devel
sudo yum install opencl-headers ocl-icd-devel
```

## Compilation

### Windows:
```cmd
compile_distributed.bat
```

### Linux/Mac:
```bash
# MPI version
mpicc -o mpi_matrix mpi_matrix.c -lm

# MPI + OpenMP version
mpicc -o mpi_openmp_matrix mpi_openmp_matrix.c -fopenmp -lm

# MPI + OpenCL version
mpicc -o mpi_opencl_matrix mpi_opencl_matrix.c -lOpenCL -lm
```

## Usage

### MPI Implementation
```bash
mpiexec -n [processes] ./mpi_matrix [matrix_size]
```
- `processes`: Number of MPI processes
- `matrix_size`: Size of N×N matrix (default: 1000)

**Example:**
```bash
mpiexec -n 4 ./mpi_matrix 1500
```

### MPI + OpenMP Hybrid
```bash
mpiexec -n [processes] ./mpi_openmp_matrix [matrix_size] [threads]
```
- `processes`: Number of MPI processes
- `matrix_size`: Size of N×N matrix (default: 1000)
- `threads`: OpenMP threads per process (default: 4)

**Example:**
```bash
mpiexec -n 2 ./mpi_openmp_matrix 1000 8
```

### MPI + OpenCL Hybrid
```bash
mpiexec -n [processes] ./mpi_opencl_matrix [matrix_size]
```
- `processes`: Number of MPI processes
- `matrix_size`: Size of N×N matrix (default: 1000)

**Example:**
```bash
mpiexec -n 2 ./mpi_opencl_matrix 2000
```

## Implementation Details

### 1. MPI Implementation
- **Strategy**: Row-wise matrix decomposition
- **Communication**: Scatter A rows, broadcast B matrix, gather C results
- **Load Balancing**: Static distribution with remainder handling
- **Synchronization**: Implicit via MPI collective operations

### 2. MPI + OpenMP Hybrid
- **MPI Level**: Distribute work across nodes/processes
- **OpenMP Level**: Parallelize computation within each process
- **Memory Model**: Distributed + shared memory hierarchy
- **Scalability**: Two-level parallelism for better resource utilization

### 3. MPI + OpenCL Hybrid
- **MPI Level**: Distribute work across nodes
- **OpenCL Level**: GPU acceleration within each node
- **Device Selection**: Automatic GPU/CPU fallback
- **Memory Management**: Host-device data transfers optimized

## Performance Characteristics

### Expected Speedup Patterns:

#### Pure MPI:
- **Good for**: CPU-bound workloads across multiple nodes
- **Scaling**: Linear up to memory bandwidth limits
- **Overhead**: Communication costs increase with process count

#### MPI + OpenMP:
- **Good for**: Multi-core nodes with shared memory
- **Scaling**: Better efficiency than pure MPI on single node
- **Overhead**: Reduced inter-process communication

#### MPI + OpenCL:
- **Good for**: GPU-accelerated clusters
- **Scaling**: Excellent for compute-intensive operations
- **Overhead**: GPU memory transfers, kernel launch costs

### Matrix Size Impact:
- **Small (N < 500)**: Communication overhead dominates
- **Medium (N = 500-1500)**: Good balance of computation/communication
- **Large (N > 1500)**: Best parallel efficiency

## Testing and Evaluation

### Automated Testing:
```bash
# Linux/Mac
chmod +x test_distributed.sh
./test_distributed.sh

# Analysis
python3 analyze_distributed.py
```

### Manual Performance Testing:
```bash
# Test different configurations
for size in 500 1000 1500; do
    for procs in 1 2 4; do
        echo "Testing N=$size, P=$procs"
        mpiexec -n $procs ./mpi_matrix $size
        mpiexec -n $procs ./mpi_openmp_matrix $size 4
    done
done
```

## Output Files

Each implementation generates:
- **Console Output**: Timing and configuration information
- **Result Files**:
  - `output_mpi.txt`
  - `output_mpi_openmp.txt`
  - `output_mpi_opencl.txt`

## Performance Analysis

The analysis script generates:
- `distributed_performance_analysis.png`: Visual performance comparison
- `distributed_performance_report.txt`: Detailed statistical analysis
- `distributed_results.csv`: Raw performance data

## Troubleshooting

### Common Issues:

#### MPI not found:
```bash
# Check MPI installation
which mpicc
mpiexec --version
```

#### OpenCL compilation fails:
- Install GPU vendor SDK
- Use CPU-only OpenCL runtime
- Skip OpenCL implementation if not available

#### Poor performance:
- Check process placement: `mpiexec --bind-to core`
- Verify NUMA topology: `numactl --hardware`
- Monitor network utilization for multi-node runs

#### Memory issues:
- Reduce matrix size for available memory
- Check swap usage: `free -h`
- Use memory-efficient algorithms for very large matrices

## Assignment Deliverables

This implementation provides:
- ✅ MPI matrix multiplication source code
- ✅ MPI + OpenMP hybrid implementation
- ✅ MPI + OpenCL hybrid implementation  
- ✅ Performance evaluation framework
- ✅ Comparative analysis tools
- ✅ Documentation and usage guides

## Performance Expectations

### Typical Results:
- **MPI**: 2-4x speedup with 4 processes
- **MPI+OpenMP**: 4-8x speedup with 2 processes × 4 threads
- **MPI+OpenCL**: 10-50x speedup with GPU acceleration (problem size dependent)

### Scalability Limits:
- **Communication Bound**: Small matrices, many processes
- **Memory Bound**: Large matrices, limited RAM
- **Compute Bound**: Optimal performance region

The implementations demonstrate different parallel programming paradigms and their trade-offs in distributed computing environments.
