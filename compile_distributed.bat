@echo off
echo Compiling Distributed Matrix Multiplication Programs...
echo.

REM Check for MPI compiler
where mpicc >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: mpicc not found. Please install Microsoft MPI or MPICH.
    echo Download from: https://www.microsoft.com/en-us/download/details.aspx?id=57467
    pause
    exit /b 1
)

REM Compile MPI version
echo Compiling MPI version...
mpicc -o mpi_matrix.exe mpi_matrix.c -lm
if %errorlevel% neq 0 (
    echo Error compiling MPI version
    pause
    exit /b 1
)

REM Compile MPI+OpenMP version
echo Compiling MPI+OpenMP version...
mpicc -o mpi_openmp_matrix.exe mpi_openmp_matrix.c -fopenmp -lm
if %errorlevel% neq 0 (
    echo Error compiling MPI+OpenMP version
    pause
    exit /b 1
)

REM Compile MPI+OpenCL version (may fail if OpenCL not available)
echo Compiling MPI+OpenCL version...
mpicc -o mpi_opencl_matrix.exe mpi_opencl_matrix.c -lOpenCL -lm
if %errorlevel% neq 0 (
    echo Warning: MPI+OpenCL compilation failed (OpenCL may not be available)
    echo This is normal if you don't have OpenCL SDK installed
)

echo.
echo Compilation completed!
echo.

REM Run sample tests
echo Running sample tests...
echo.

echo Testing MPI (500x500 matrix, 2 processes):
mpiexec -n 2 mpi_matrix.exe 500
echo.

echo Testing MPI+OpenMP (500x500 matrix, 2 processes, 2 threads):
mpiexec -n 2 mpi_openmp_matrix.exe 500 2
echo.

if exist mpi_opencl_matrix.exe (
    echo Testing MPI+OpenCL (500x500 matrix, 2 processes):
    mpiexec -n 2 mpi_opencl_matrix.exe 500
    echo.
)

echo.
echo Sample tests completed!
echo Check output files: output_mpi.txt, output_mpi_openmp.txt, output_mpi_opencl.txt
echo.
pause