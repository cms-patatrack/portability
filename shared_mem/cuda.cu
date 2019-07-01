#include <cuda_runtime.h>


__device__
void positive(float x, int &counter) {
  if (x > 0.) {
    atomicAdd(&counter, 1);
  }
}




__global__
void kernel(float *data, int *counter, unsigned int size) {
  __shared__ int shared;
  if (threadIdx.x == 0) {
    shared = 0;
  }

  __syncthreads();

  auto first_thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto grid_size    = gridDim.x * blockDim.x;
  for (auto i = first_thread; i < size; i += grid_size) {
    positive(data[i], shared);
  }




  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(counter, shared);
  }
}



#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

const unsigned int size = 9999;

int main() {
  int counter = 0;
  std::vector<float> data(size, 0.);

  // generate the input data
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (auto &x : data)
    x = distribution(generator);

  float *data_d;
  cudaMalloc(&data_d, size * sizeof(float));
  cudaMemcpy(data_d, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  int *counter_d;
  cudaMalloc(&counter_d, sizeof(int));
  cudaMemset(counter_d, 0, sizeof(int));

  kernel<<<32, 32>>>(data_d, counter_d, size);

  cudaMemcpy(&counter, counter_d, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << counter << std::endl;
}
