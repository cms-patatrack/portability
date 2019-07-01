#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <SYCL/sycl.hpp>

#include "../common/sycl_common.h"


// declare the SYCL kernel name in the global namespace
class vector_add;

int main() {
  // list SYCL devices
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

  // beginning of SYCL objects' scope
  {
    std::cout << "Entering SYCL object scope" << std::endl;

    // select a device supported by the backend being used
    auto device = select_backend_device(std::cout);

    // construct a SYCL queue for the chosen device
    auto queue = cl::sycl::queue(device);

    // SYCL buffers
    auto A_buff = cl::sycl::buffer<float>(A.data(), array_size);
    auto B_buff = cl::sycl::buffer<float>(B.data(), array_size);
    auto C_buff = cl::sycl::buffer<float>(C.data(), array_size);

    // get the number of compute units, threads per block, and determine the total number of threads
    auto block_size    = device.get_info<cl::sycl::info::device::max_work_group_size>();
    auto num_blocks    = device.get_info<cl::sycl::info::device::max_compute_units>();
    auto total_threads = std::min<size_t>(block_size * num_blocks, array_size);

    // submit the kernel to the queue
    queue.submit([&](cl::sycl::handler &cgh) {
      // get read and write access to the SYCL buffers inside the device kernel
      std::cout << "Register read and write access to the SYCL buffers" << std::endl;
      auto A_acc = A_buff.get_access<cl::sycl::access::mode::read>(cgh);
      auto B_acc = B_buff.get_access<cl::sycl::access::mode::read>(cgh);
      auto C_acc = C_buff.get_access<cl::sycl::access::mode::write>(cgh);

      // launch the kernel
      std::cout << "Launch kernel" << std::endl;
      std::cout << "  block size: "    << block_size    << std::endl;
      std::cout << "  grid size:  "    << num_blocks    << std::endl;
      std::cout << "  total threads: " << total_threads << std::endl;
      cgh.parallel_for<class vector_add>(cl::sycl::range<1>{total_threads}, [=](cl::sycl::item<1> itemId) {
        // local thread id, thread size, and data size
        auto thread_id     = itemId.get_id(0);
        auto total_threads = itemId.get_range()[0];
        auto array_size    = C_acc.get_count();
        // loop over the input data
        for (auto i = thread_id; i < array_size; i += total_threads) {
          C_acc[i] = A_acc[i] + B_acc[i];
        }
      });
    });

    std::cout << "Leaving SYCL object scope" << std::endl;
  }
  // end of SYCL objects' scope

  // validate the results
  int incorrect = 0;
  for (int i = 0; i < array_size; ++i) {
    if (C[i] != A[i] + B[i])
      ++incorrect;
  }
  if (incorrect) {
    std::cout << "Found " << incorrect << " incorrect results out of " << array_size << std::endl;
  } else {
    std::cout << "All results are correct" << std::endl;
  }

  return EXIT_SUCCESS;
}
