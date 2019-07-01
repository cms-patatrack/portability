#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include "../common/cuda_common.h"

// CUDA device kernel
__global__
void vector_add(const float *A, const float *B, float *C, size_t array_size) {
  // local thread id, thread size, and data size
  auto thread_id     = threadIdx.x + blockDim.x * blockIdx.x;
  auto total_threads = blockDim.x * gridDim.x;
  // loop over the input data
  for (size_t i = thread_id; i < array_size; i += total_threads) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  // list CUDA devices
  list_devices(std::cout);
  std::cout << std::endl;

  // generator for the input data
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);

  // array size
  const int array_size = 1024*1024;

  // input and output host data
  std::vector<float> A(array_size);
  std::vector<float> B(array_size);
  std::vector<float> C(array_size);

  // generate the input data
  std::cout << "Generate input data" << std::endl;
  for (auto &x : A)
    x = distribution(generator);
  for (auto &x : B)
    x = distribution(generator);

  // allocate device memory
  float *A_dev;
  float *B_dev;
  float *C_dev;
  cudaMalloc((void **)&A_dev, array_size * sizeof(float));
  cudaMalloc((void **)&B_dev, array_size * sizeof(float));
  cudaMalloc((void **)&C_dev, array_size * sizeof(float));

  // copy the data from the host to the device
  std::cout << "Copy the data to the device" << std::endl;
  cudaMemcpy(A_dev, A.data(), array_size * sizeof(float), cudaMemcpyDefault);
  cudaMemcpy(B_dev, B.data(), array_size * sizeof(float), cudaMemcpyDefault);

  // get the number of compute units, threads per block, and determine the total number of threads
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, 0);
  //auto block_size = std::min<int>(getCudaCoresPerSM(properties.major, properties.minor), array_size);
  auto block_size = std::min<int>(properties.maxThreadsPerBlock, array_size);
  auto num_blocks = std::min<int>(properties.multiProcessorCount, (array_size + block_size - 1) / block_size);
  auto total_threads = block_size * num_blocks;

  // launch the kernel
  std::cout << "Launch kernel" << std::endl;
  std::cout << "  block size: "    << block_size    << std::endl;
  std::cout << "  grid size:  "    << num_blocks    << std::endl;
  std::cout << "  total threads: " << total_threads << std::endl;
  vector_add<<<num_blocks, block_size>>>(A_dev, B_dev, C_dev, array_size);
  cudaDeviceSynchronize();

  // return the result to the host vector
  std::cout << "Copy the results to the host" << std::endl;
  cudaMemcpy(C.data(), C_dev, array_size * sizeof(float), cudaMemcpyDefault);

  // validate the results
  std::cout << "Validate the results" << std::endl;
  int incorrect = 0;
  for (int i = 0; i < array_size; ++i) {
    if (C[i] != A[i] + B[i]) {
      ++incorrect;
      std::cout << A[i] << " + " << B[i] << " != " << C[i] << std::endl;
    }
  }
  if (incorrect) {
    std::cout << "Found " << incorrect << " incorrect results out of " << array_size << std::endl;
  } else {
    std::cout << "All results are correct" << std::endl;
  }

  // release the memory objects
  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);

  return EXIT_SUCCESS;
}
