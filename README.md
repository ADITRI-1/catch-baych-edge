# CUDA Batch Sobel Edge Detection

Process 512 distinct synthetic grayscale images (256 x 256) using a custom CUDA
Sobel kernel. This is 33,554,432 pixels. A CPU implementation checks every GPU
output pixel; it is a reference, not a substitute for GPU computation.

## Requirements

Linux, NVIDIA GPU and working driver, CUDA toolkit with nvcc, compatible C++
compiler, GNU Make, Bash, and Python 3 (standard library only). No OpenCV, NPP,
NumPy or image downloads are needed. This project requires a GPU and fails if
CUDA is unavailable. It does not silently fall back to CPU.

## Build and run

```bash
make build
bash scripts/run.sh
```

The script records hardware/compiler information, processes the dataset,
checks correctness, and creates `execution_evidence.zip` only after a successful
run. Expected completion marker: `Validation: PASS`, with zero pixel mismatches.
This is an expected marker, not a claim that execution has already occurred.
To repeat, rename `results` before running the script again.

Custom execution (create the directory first):

```bash
mkdir -p custom_results
./edge_detect --count 600 --width 256 --height 256 --batch 20 --seed 7 --output custom_results
./edge_detect --help
```

Options: count 1..100000; width/height 3..8192; batch 1..1024; integer seed;
existing output directory. Use hundreds of images for the assignment. Each batch
is capped at 512 MiB per array; reduce batch size for limited device/host memory.
The final partial batch is supported. Default host arrays use about 6 MiB and
device buffers 4 MiB. The default input/output images use about 64 MiB on disk.

## Algorithm and GPU work

A 16 x 16 thread block processes pixels; grid.z identifies the image within a
batch. Each thread reads a 3 x 3 neighborhood and computes horizontal and
vertical Sobel gradients. Output is min(255, abs(Gx) + abs(Gy)); this is L1
magnitude, not Euclidean magnitude. Borders are explicitly zero. Independent
output pixels require no atomics or cross-block synchronization. Two device
buffers are allocated once and reused. Batches bound memory usage.

## Dataset and reproducibility

The program generates original synthetic images with shifted stripe patterns,
rectangles, random-position disks and small noise using a seeded MT19937.
The default seed is 42. All 512 input images and GPU outputs are saved in P5 PGM
format. Synthetic data is used transparently, not represented as photographs or
an external benchmark. No external dataset license is needed. The generator is
part of the C++ source and allows offline reproduction.

## Validation and timing

Every pixel is compared exactly against the integer CPU Sobel reference,
including border pixels. Any mismatch returns exit code 2. CUDA/I/O/argument
errors return nonzero. Warm-up work is excluded from measurements.
`metrics.csv` records per-batch kernel time using CUDA events, wall time for
host-to-device transfer + kernel + device-to-host transfer, CPU reference time,
and mismatch counts. Dataset generation, allocation, validation comparison,
and disk writes are excluded from those measured intervals. CPU timing includes
zeroing the reference buffer. These are simple single-run measurements, not a
rigorous benchmark; no speedup is promised. Transfer overhead can dominate for
small inputs. All result images are produced by the actual GPU kernel.

## Evidence and submission

After a successful lab run:

- `results/environment.txt`: date, CUDA compiler and GPU details.
- `results/run.log`: actual processing progress, timings and validation.
- `results/metrics.csv`: per-batch timing and correctness records.
- `results/input_*.pgm`: original images, before processing.
- `results/gpu_edges_*.pgm`: corresponding GPU results, after processing.
- `execution_evidence.zip`: compressed evidence for Coursera upload.

PGM images can be opened in image viewers supporting Netpbm, such as GIMP or
ImageMagick. Include source, Makefile, scripts, README and results in a public
repository, and provide its main-page URL. Check that the URL opens while logged
out. Upload execution_evidence.zip separately as proof. Use DESCRIPTION.md as a
starting description and add real measurements after running.

Do not claim successful GPU execution until your lab run finishes. The delivered
source package does not include fabricated logs or benchmark numbers.

## Limitations and improvements

Synthetic patterns demonstrate scale and correctness but not performance on a
real photographic dataset. The current program generates data rather than
loading arbitrary photos. Transfers are synchronous; future work could add
pinned buffers, streams, shared-memory tiling and repeated benchmark trials.
The CPU reference is scalar. Code uses descriptive functions, two-space
indentation and explicit error handling inspired by Google C++ style; formal
full style compliance has not been certified.

## Reference

NVIDIA CUDA Programming Guide, asynchronous execution and event timing:
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
"# catch-baych-edge" 
