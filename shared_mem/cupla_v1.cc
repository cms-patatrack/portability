#include <cuda_to_cupla.hpp>

template <typename T_Acc>
ALPAKA_FN_ACC
void positive(T_Acc const& acc, float x, int &counter) {
  if (x > 0.) {
    atomicAdd(&counter, 1);
  }
}

struct kernel {

template <typename T_Acc>
ALPAKA_FN_ACC
void operator()(T_Acc const& acc, float *data, int *counter, unsigned int size) const {
  sharedMem(shared, int);
  if (threadIdx.x == 0) {
    shared = 0;
  }

  __syncthreads();

  auto first_thread = threadIdx.x + blockIdx.x * blockDim.x;
  auto grid_size    = gridDim.x * blockDim.x;
  for (auto i = first_thread; i < size; i += grid_size) {
    positive(acc, data[i], shared);
  }




  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(counter, shared);
  }
}

};

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
  cudaMalloc((void **) &data_d, size * sizeof(float));
  cudaMemcpy(data_d, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

  int *counter_d;
  cudaMalloc((void **) &counter_d, sizeof(int));
  cudaMemset(counter_d, 0, sizeof(int));

  CUPLA_KERNEL_OPTI(kernel)(32, 32)(data_d, counter_d, size);

  cudaMemcpy(&counter, counter_d, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << counter << std::endl;
}
