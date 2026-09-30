#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void Check(cudaError_t result) {
  if (result != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(result));
  }
}

// Integer L1 Sobel magnitude, saturated to 8 bits. Borders are zero.
__global__ void Sobel(const unsigned char* input, unsigned char* output,
                      int width, int height, int count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int image = blockIdx.z;
  if (x >= width || y >= height || image >= count) return;
  size_t i = static_cast<size_t>(image) * width * height + y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[i] = 0;
    return;
  }
  int gx = -input[i-width-1] + input[i-width+1]
           -2*input[i-1] + 2*input[i+1]
           -input[i+width-1] + input[i+width+1];
  int gy = -input[i-width-1] - 2*input[i-width] - input[i-width+1]
           +input[i+width-1] + 2*input[i+width] + input[i+width+1];
  int magnitude = abs(gx) + abs(gy);
  output[i] = static_cast<unsigned char>(magnitude > 255 ? 255 : magnitude);
}

void CpuSobel(const std::vector<unsigned char>& input,
              std::vector<unsigned char>* output, int w, int h, int count) {
  std::fill(output->begin(), output->end(), 0);
  for (int image = 0; image < count; ++image) {
    for (int y = 1; y < h - 1; ++y) {
      for (int x = 1; x < w - 1; ++x) {
        size_t i = static_cast<size_t>(image) * w * h + y * w + x;
        int gx = -input[i-w-1] + input[i-w+1] - 2*input[i-1]
                 +2*input[i+1] - input[i+w-1] + input[i+w+1];
        int gy = -input[i-w-1] - 2*input[i-w] - input[i-w+1]
                 +input[i+w-1] + 2*input[i+w] + input[i+w+1];
        (*output)[i] = static_cast<unsigned char>(
            std::min(255, std::abs(gx) + std::abs(gy)));
      }
    }
  }
}

void WritePgm(const std::string& path, const unsigned char* pixels, int w, int h) {
  std::ofstream file(path, std::ios::binary);
  file << "P5\n" << w << " " << h << "\n255\n";
  file.write(reinterpret_cast<const char*>(pixels), static_cast<size_t>(w)*h);
  if (!file) throw std::runtime_error("Could not write " + path);
}

int main(int argc, char** argv) {
  try {
    int count = 512, width = 256, height = 256, batch = 32, seed = 42;
    std::string out = "results";
    for (int a = 1; a < argc; ++a) {
      std::string key = argv[a];
      if (key == "--help") {
        std::cout << "Options: --count N --width W --height H --batch N "
                     "--seed N --output EXISTING_DIRECTORY\n";
        return 0;
      }
      if (a + 1 == argc) throw std::runtime_error("Missing option value");
      std::string value = argv[++a];
      if (key == "--output") { out = value; continue; }
      size_t used = 0;
      int n = std::stoi(value, &used);
      if (used != value.size()) throw std::runtime_error("Invalid integer");
      if (key == "--count") count = n;
      else if (key == "--width") width = n;
      else if (key == "--height") height = n;
      else if (key == "--batch") batch = n;
      else if (key == "--seed") seed = n;
      else throw std::runtime_error("Unknown option: " + key);
    }
    if (count < 1 || count > 100000 || width < 3 || height < 3 ||
        width > 8192 || height > 8192 || batch < 1 || batch > 1024) {
      throw std::runtime_error("Invalid dimensions/count/batch; see README");
    }
    batch = std::min(batch, count);
    size_t plane = static_cast<size_t>(width)*height;
    size_t capacity = plane*batch;
    if (capacity > 512ULL*1024*1024) {
      throw std::runtime_error("Batch exceeds 512 MiB; reduce --batch");
    }
    Check(cudaSetDevice(0));
    cudaDeviceProp props;
    Check(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << "\nImages: " << count
              << "\nDimensions: " << width << "x" << height
              << "\nBatch: " << batch << "\nSeed: " << seed << "\n";
    std::vector<unsigned char> input(capacity), gpu(capacity), cpu(capacity);
    unsigned char *device_input = nullptr, *device_output = nullptr;
    Check(cudaMalloc(reinterpret_cast<void**>(&device_input), capacity));
    Check(cudaMalloc(reinterpret_cast<void**>(&device_output), capacity));
    cudaEvent_t start, stop;
    Check(cudaEventCreate(&start));
    Check(cudaEventCreate(&stop));
    dim3 threads(16, 16);
    // Warm up runtime and kernel; exclude this work from recorded timings.
    Check(cudaMemset(device_input, 0, capacity));
    Sobel<<<dim3((width+15)/16, (height+15)/16, batch), threads>>>(
        device_input, device_output, width, height, batch);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    std::ofstream csv(out + "/metrics.csv");
    if (!csv) throw std::runtime_error("Create the output directory first");
    csv << "first_image,count,kernel_ms,gpu_transfer_compute_ms,cpu_ms,mismatches\n";
    std::mt19937 rng(static_cast<uint32_t>(seed));
    double total_kernel = 0, total_gpu = 0, total_cpu = 0;
    size_t mismatches = 0;
    using Clock = std::chrono::steady_clock;
    for (int first = 0; first < count; first += batch) {
      int current = std::min(batch, count-first);
      size_t bytes = plane*current;
      // Original synthetic images: changing rectangles, disks, stripe patterns
      // and bounded noise. No external data download or license required.
      for (int k = 0; k < current; ++k) {
        int id = first+k;
        int cx = width/4 + static_cast<int>(rng() % (width/2));
        int cy = height/4 + static_cast<int>(rng() % (height/2));
        int radius = std::max(1, std::min(width,height)/8);
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            int value = 25 + ((x + id*3) % 64)/4;
            if (x > width/8 && x < width/2 && y > height/5 && y < height/2)
              value = 170;
            if ((x-cx)*(x-cx)+(y-cy)*(y-cy) < radius*radius) value = 220;
            if (y > 3*height/4) value = ((x+id) / 8 % 2) ? 190 : 50;
            input[static_cast<size_t>(k)*plane+y*width+x] =
                static_cast<unsigned char>(value + rng()%9);
          }
        }
      }
      auto wall_start = Clock::now();
      Check(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
      Check(cudaEventRecord(start));
      Sobel<<<dim3((width+15)/16, (height+15)/16, current), threads>>>(
          device_input, device_output, width, height, current);
      Check(cudaGetLastError());
      Check(cudaEventRecord(stop));
      Check(cudaEventSynchronize(stop));
      float kernel_ms = 0;
      Check(cudaEventElapsedTime(&kernel_ms, start, stop));
      Check(cudaMemcpy(gpu.data(), device_output, bytes, cudaMemcpyDeviceToHost));
      double gpu_ms = std::chrono::duration<double,std::milli>(
          Clock::now()-wall_start).count();
      auto cpu_start = Clock::now();
      CpuSobel(input, &cpu, width, height, current);
      double cpu_ms = std::chrono::duration<double,std::milli>(
          Clock::now()-cpu_start).count();
      size_t errors = 0;
      for (size_t i = 0; i < bytes; ++i) errors += gpu[i] != cpu[i];
      mismatches += errors;
      total_kernel += kernel_ms; total_gpu += gpu_ms; total_cpu += cpu_ms;
      csv << first << ',' << current << ',' << kernel_ms << ',' << gpu_ms
          << ',' << cpu_ms << ',' << errors << '\n';
      for (int k = 0; k < current; ++k) {
        std::string name = std::to_string(first+k);
        WritePgm(out+"/input_"+name+".pgm", input.data()+k*plane,width,height);
        WritePgm(out+"/gpu_edges_"+name+".pgm",gpu.data()+k*plane,width,height);
      }
      std::cout << "Processed " << first+current << "/" << count
                << "; mismatches=" << errors << '\n';
    }
    csv.close();
    if (!csv) throw std::runtime_error("Failed to save metrics");
    Check(cudaEventDestroy(start)); Check(cudaEventDestroy(stop));
    Check(cudaFree(device_input)); Check(cudaFree(device_output));
    std::cout << "Total pixels: " << plane*count
              << "\nKernel total ms: " << total_kernel
              << "\nGPU transfers+compute total ms: " << total_gpu
              << "\nCPU reference total ms: " << total_cpu
              << "\nPixel mismatches: " << mismatches
              << "\nValidation: " << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
    return mismatches == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
