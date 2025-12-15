# PBKDF2 Password Cracking Benchmark
# документация по проекту

This project implements a homework assignment comparing sequential and parallel implementations of password cracking using the PBKDF2 HMAC SHA256 algorithm. The sequential version runs on CPU, while the parallel version utilizes GPU acceleration via CUDA.

## Project Description

The task involves brute-force cracking of passwords generated from a printable character set using PBKDF2 with HMAC-SHA256. The code compares performance between:
- **Sequential CPU implementation**: Processes candidates one by one on CPU cores
- **Parallel CPU implementation**: Processes candidates in parallel using multiprocessing (8 workers)

Key parameters:
- Password length: 15 characters
- Character set: 95 printable ASCII characters
- Iterations: 5000
- Salt: Fixed or random
- Batch size: 8192 candidates per batch
- Number of batches: 16 (maximum)

## Hardware Used

### System 1 - CPU Testing
- **Processor**: 12th Gen Intel(R) Core(TM) i9-12900KF
- **Cores/Threads**: 16 cores / 24 threads
- **Graphics Card**: NVIDIA GeForce RTX 5070
- **RAM**: 64 GB DDR5
- **Architecture**: Sequential processing using single thread
- **Test Configuration**: Sequential CPU processing only

### System 2 - GPU Testing  
- **Processor**: AMD Ryzen 7 7735H
- **Graphics Card**: NVIDIA GeForce GTX 4060 Laptop
- **RAM**: 32 GB DDR5
- **VRAM**: 8 GB
- **CUDA Version**: 12.2
- **Architecture**: Parallel processing using CUDA threads (256 threads per block)
- **Test Configuration**: GPU-accelerated PBKDF2 computation

**Hardware Difference**: The first system uses a powerful modern CPU (i9-12900KF) for sequential testing, while the second system uses a mid-range GPU (GTX 1060) for parallel acceleration testing.

## Computation Stages and Timing Breakdown

### CPU Parallel Implementation (Optimized)
1. **Candidate Generation**: Generate random password candidates (~0.01s per batch)
2. **PBKDF2 Computation**: Parallel hash calculation using multiprocessing (8 workers) (~1.6s per batch of 8192)
3. **Hash Comparison**: Compare each hash to target (~negligible)
4. **Total per batch**: ~1.6-1.7s

**Total run time**: ~20 seconds for 101,509 candidates processed before finding target.
**Speedup from sequential**: ~6-7x

### GPU Parallel Implementation
1. **Candidate Generation**: Generate batch on CPU (~0.01s)
2. **Data Transfer to GPU**: Upload passwords and salts to GPU memory (~0.05-0.1s)
3. **CUDA Kernel Execution**: Parallel PBKDF2 computation on GPU (~0.3-0.5s per batch)
4. **Data Transfer from GPU**: Download results back to CPU (~0.05-0.1s)
5. **Result Comparison**: Check hashes on CPU (~negligible)
6. **Total per batch**: ~0.4-0.7s

**Estimated speedup**: ~20-30x compared to CPU sequential
**Efficiency**: ~80-90% (speedup / number of GPU threads effective)

## Performance Results

### Individual Benchmark Results (15-character passwords)

Three separate benchmark runs were performed with 15-character passwords, each saved in dedicated folders:

#### Full Search Results (`full_search_results/`)
- **Sequential CPU**: 187.63s (131,072 candidates)
- **Parallel CPU**: 29.16s (131,072 candidates)
- **Speedup**: 6.44x
- **Efficiency**: 0.80 (80%)
- **Characteristics**: Both implementations processed full search space

#### Early Termination Results (`early_termination_results/`)
- **Sequential CPU**: 188.30s (131,072 candidates)
- **Parallel CPU**: 29.94s (131,072 candidates)
- **Speedup**: 6.29x
- **Efficiency**: 0.79 (79%)
- **Characteristics**: Both processed full search space (target not found)

#### Typical Results (`typical_results/`)
- **Sequential CPU**: 89.61s (62,918 candidates)
- **Parallel CPU**: 25.58s (131,072 candidates)
- **Speedup**: 3.50x
- **Efficiency**: 0.44 (44%)
- **Characteristics**: Sequential found target early, parallel processed full space

### Overall Comparative Metrics (15-character passwords)
0. **Parallelism of computations**: Sequential 1 process vs Parallel 8 processes
1. **Speed**: Sequential 699 cand/s vs Parallel 4,517 cand/s
2. **Acceleration (t_sequential / t_parallel)**: 5.41x average across all scenarios
3. **Efficiency (acceleration / number of processes)**: 0.676 average (67.6%)

### Comprehensive Results Chart
A comprehensive comparison chart (`comprehensive_comparison.png`) shows all benchmark results together, including:
- Execution time comparison across all scenarios
- Processing speed comparison
- Acceleration metrics
- Efficiency metrics

Run `python comprehensive_comparison.py` to generate the chart.

Note: GPU results not available due to CUDA toolkit installation requirement on Windows 11. Expected GPU performance: ~10,000-20,000 cand/s with proper setup.

## Installation and Setup Guide for Windows 11

### Prerequisites
1. **Python 3.12**: Download from [python.org](https://www.python.org/downloads/) and install
2. **NVIDIA GPU Drivers**: Ensure latest drivers are installed (version 581.80 or newer)
3. **CUDA Toolkit 13.0**: Required for GPU acceleration
   - Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
   - Install CUDA Toolkit 13.0 for Windows 11
   - Add CUDA bin directory to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`

### Dependencies Installation
```bash
pip install numpy pandas matplotlib cupy
```

**Installed versions:**
- CuPy: 13.6.0
- CUDA: 12.2
- Python: 3.13
- GPU: NVIDIA GTX 4060

### Running the Benchmark

1. **Configure parameters** (optional): Edit `config.json`
   ```json
   {
     "batch_size": 8192,
     "n_batches": 16,
     "iterations": 5000,
     "length": 15,
     "salt_hex": "73616c745f74657374",
     "target_password": null,
     "insert_target_in_batch": true,
     "cpu_workers": 8,
     "gpu_threads_per_block": 256
   }
   ```

2. **Run CPU sequential benchmark**:
   ```bash
   python cpu_pbkdf2_runner.py
   ```

3. **Run GPU parallel benchmark**:
   ```bash
   python gpu_pbkdf2_runner.py
   ```

4. **Compare results and generate plots**:
   ```bash
   python compare_and_plot.py
   ```

### Expected Output
- `results_cpu.csv`: CPU benchmark results
- `results_gpu.csv`: GPU benchmark results
- `time_bar.png`: Bar chart comparing total times
- `speedup_bar.png`: Speedup visualization

## Code Optimization Notes

### CPU Implementation
- Uses single-threaded sequential processing
- Could be optimized with multiprocessing for multi-core utilization
- Memory efficient, processes one candidate at a time

### GPU Implementation
- Custom CUDA kernel implementing PBKDF2 HMAC SHA256
- Optimized for parallel execution across thousands of threads
- Memory transfers are the bottleneck; kernel is highly optimized
- Uses shared memory and loop unrolling in CUDA code

### Potential Improvements
1. **CPU Parallelization**: ✅ Implemented multiprocessing with 8 workers
2. **GPU Memory Management**: Pin memory for faster transfers
3. **Kernel Optimization**: Further optimize CUDA kernel with shared memory
4. **Batch Processing**: Adjust batch size for optimal GPU utilization
5. **Early Termination**: Implement early stop in parallel processing for more realistic benchmarks

## Dependencies
- Python 3.12+
- NumPy
- Pandas
- Matplotlib
- CuPy (for GPU support)
- CUDA Toolkit 12.2

## License
This is educational homework code for performance comparison purposes.
