#include <SYCL/sycl.hpp>

#include "../common/sycl_common.h"

inline
void positive(float x, int & counter) {
  if (x > 0.) {
    cl::sycl::atomic_fetch_add(cl::sycl::atomic<int, cl::sycl::access::address_space::local_space>(cl::sycl::local_ptr<int>(&counter)), 1);
  }
}

// declare the SYCL kernel name in the global namespace
struct kernel {
public:
  using data_accessor_t = cl::sycl::accessor<float, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;
  using counter_accessor_t = cl::sycl::accessor<int, 1, cl::sycl::access::mode::atomic, cl::sycl::access::target::global_buffer>;

  kernel(data_accessor_t data_, counter_accessor_t counter_) :
    data{std::move(data_)},
    counter{std::move(counter_)}
  {}

  void operator() (cl::sycl::group<1> group) {
    // data size
    const auto data_size = data.get_count();

    // per-workgroup shared memory
    int shared = 0;

    // implicit workgroup barrier

    // loop over the input data
    group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
      for (auto i = item.get_global_id(0); i < data_size; i += item.get_global_range(0))
        positive(data[i], shared);
    });

    // implicit workgroup barrier

    // according to the SYCL 1.2.1 specifications (4.8.5.3, pages 173-176), this
    // should be executed only once per workgroup:
    // 
    //    The body of the outer parallel_for_work_group call consists of a lambda
    //    function or function object. The body of this function object contains
    //    code that is executed only once for the entire work-group. If the code
    //    has no side-effects and the compiler heuristic suggests that it is more
    //    efficient to do so, this code will be executed for each work-item.
    //
    // The workgroup code clearly has side-effects (the atomic_fetch_add call), so
    // it should always run once per workgroup.
    // 
    // Unfortunately, at least using an Intel(R) Gen9 HD Graphics NEO OpenCL backend,
    // it executes for each thread, resulting in the wrong results.
#ifdef SYCL_WORKAROUND_WORK_ITEM_LOOP_BUG
    group.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
      if (item.get_local_id(0) == 0)
#endif // SYCL_WORKAROUND_WORK_ITEM_LOOP_BUG
        cl::sycl::atomic_fetch_add(counter[0], shared);
#ifdef SYCL_WORKAROUND_WORK_ITEM_LOOP_BUG
    });
#endif // SYCL_WORKAROUND_WORK_ITEM_LOOP_BUG
  }

private:
  data_accessor_t data;
  counter_accessor_t counter;
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

  // beginning of SYCL objects' scope
  {
    // select a device supported by the backend being used
    cl::sycl::device device = select_backend_device(std::cout);

    // get the number of compute units, threads per block, and determine the total number of threads
    size_t block_size = device.get_info<cl::sycl::info::device::max_work_group_size>();
    size_t grid_size  = (size + block_size - 1) / block_size;

    // constructing a SYCL queue for the selected device
    auto queue = cl::sycl::queue(device);

    // SYCL buffers
    auto data_buf = cl::sycl::buffer<float>(data.data(), size);
    auto counter_buf = cl::sycl::buffer<int>(& counter, 1);

    // submit the kernel to the queue
    queue.submit([&](cl::sycl::handler &cgh) {
      // give access to the SYCL buffers to the device kernel
      auto data_acc    = data_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto counter_acc = counter_buf.get_access<cl::sycl::access::mode::atomic>(cgh);

      // launch the kernel
      std::cout << "Launch kernel" << std::endl;
      std::cout << "  block size: " << block_size << std::endl;
      std::cout << "  grid size:  " << grid_size  << std::endl;
      cgh.parallel_for_work_group(cl::sycl::range<1>{grid_size}, cl::sycl::range<1>{block_size}, kernel(data_acc, counter_acc));
    });
  }
  // end of SYCL objects' scope

  std::cout << counter << std::endl;

  return EXIT_SUCCESS;
}
