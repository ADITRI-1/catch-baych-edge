Project title: GPU-Accelerated Batch Image Processing with CUDA Sobel Edge Detection

This project implements batch edge detection using a custom CUDA C++ kernel.
Its default workload contains 512 synthetic grayscale images of 256 x 256
pixels, representing more than 33 million pixels. The dataset contains varying
geometric shapes, stripe patterns, and noise and is generated reproducibly
without external downloads. Each GPU thread calculates the Sobel response for
one pixel, and the program processes images in batches to limit memory usage.

The project includes command-line options, a Makefile, execution scripts, and
a CPU reference that checks every GPU output pixel. It records kernel timing
separately from transfer-plus-computation timing, and saves before/after images
and CSV metrics. This design makes both the amount of data processed and the
correctness of the GPU results reviewable. The CPU implementation is used only
for validation; the submitted image-processing pipeline requires CUDA.

Before submitting, add your actual GPU name, processed-image count, validation
result and measured timings from results/run.log. Describe any issues you
actually encountered; do not claim measurements or successful execution before
running the project in the GPU laboratory.
