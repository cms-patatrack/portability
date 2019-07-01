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

  auto first_thread = (threadIdx.x + blockIdx.x * blockDim.x) * elemDim.x;
  auto grid_size    = gridDim.x * blockDim.x * elemDim.x;
  for (auto first_element = first_thread; first_element < size; first_element += grid_size) {
    auto last_element = std::min(first_element + elemDim.x, size);
    for (auto i = first_element; i < last_element; ++i) {
      positive(acc, data[i], shared);
    }
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
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  cudaMalloc((void **) &data_d, size * sizeof(float));
  cudaMemcpy(data_d, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
#else
  data_d = data.data();
#endif

  int *counter_d;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  cudaMalloc((void **) &counter_d, sizeof(int));
  cudaMemset(counter_d, 0, sizeof(int));
#else
  counter_d = &counter;
#endif

  CUPLA_KERNEL_OPTI(kernel)(32, 32)(data_d, counter_d, size);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  cudaMemcpy(&counter, counter_d, sizeof(int), cudaMemcpyDeviceToHost);
#elif CUPLA_STREAM_ASYNC_ENABLED
  // if running on the host in asynchronous mode, synchronise before reading the result
  cudaDeviceSynchronize();
#endif

  std::cout << counter << std::endl;
}
