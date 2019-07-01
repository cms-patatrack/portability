#ifndef sycl_common_h
#define sycl_common_h

#include <optional>
#include <SYCL/sycl.hpp>

// Adapted from computecpp-sdk/samples/custom-device-selector/custom-device-selector.cpp
//
// The device selection is performed via the operator() in the base selector class.
// This method will be called once per device in each platform.
// Note that all platforms are evaluated whenever there is a device selection.

class sycl_spir_selector : public cl::sycl::device_selector {
public:
  int operator()(cl::sycl::device const& device) const override {
    // require that the device supports the SPIR backend
    if (not device.has_extension("cl_khr_spir")) {
      // devices with a negative score will never be selected
      return -1;
    }

    // give higher priority to GPU devices
    if (device.get_info<cl::sycl::info::device::device_type>() == cl::sycl::info::device_type::gpu) {
      return 100;
    }

    // give lower priority to CPU devices
    if (device.get_info<cl::sycl::info::device::device_type>() == cl::sycl::info::device_type::cpu) {
      return 50;
    }

    // other devices are not supported
    return -1;
  }

};

class sycl_ptx_selector : public cl::sycl::device_selector {
public:
  int operator()(cl::sycl::device const& device) const {
    // require that the device supports the PTX backend, by requiring that it
    // supports a priprietary NVIDIA extension
    if (not device.has_extension("cl_nv_device_attribute_query")) {
      // devices with a negative score will never be selected
      return -1;
    }

    // give higher priority to GPU devices
    if (device.get_info<cl::sycl::info::device::device_type>() == cl::sycl::info::device_type::gpu) {
      return 100;
    }

    // give lower priority to CPU devices
    if (device.get_info<cl::sycl::info::device::device_type>() == cl::sycl::info::device_type::cpu) {
      return 50;
    }

    // other devices are not supported
    return -1;
  }

};


// select a device supported by the backend being used
inline
cl::sycl::device select_backend_device() {
  cl::sycl::device device;
#if defined SYCL_TARGET_HOST
  device = cl::sycl::device(cl::sycl::host_selector());
#else
  try {
#if defined SYCL_TARGET_SPIR
    device = cl::sycl::device(sycl_spir_selector());
#elif defined SYCL_TARGET_PTX
    device = cl::sycl::device(sycl_ptx_selector());
#else
    device = cl::sycl::device(cl::sycl::default_selector());
#endif // SYCL_TARGET_SPIR
  } catch(cl::sycl::exception const&) {
    device = cl::sycl::device(cl::sycl::host_selector());
  }
#endif // SYCL_TARGET_HOST
  return device;
}

// select a device supported by the backend being used
template <typename T>
cl::sycl::device select_backend_device(T & log) {
  cl::sycl::device device;
#if defined SYCL_TARGET_HOST
  log << "Running on SYCL " << device.get_info<cl::sycl::info::device::name>() << std::endl;
  device = cl::sycl::device(cl::sycl::host_selector());
#else
  try {
#if defined SYCL_TARGET_SPIR
    device = cl::sycl::device(sycl_spir_selector());
#elif defined SYCL_TARGET_PTX
    device = cl::sycl::device(sycl_ptx_selector());
#else
    device = cl::sycl::device(cl::sycl::default_selector());
#endif // SYCL_TARGET_SPIR
    log << "Running on SYCL device " << device.get_info<cl::sycl::info::device::name>() << std::endl;
  } catch(cl::sycl::exception const&) {
    log << "Falling back to SYCL " << device.get_info<cl::sycl::info::device::name>() << std::endl;
    device = cl::sycl::device(cl::sycl::host_selector());
  }
#endif // SYCL_TARGET_HOST
  return device;
}

// list available SYCL devices
template <typename T>
void list_devices(T & log) {
  auto devices = cl::sycl::device::get_devices();
  // include the fallback host device in the list
  devices.emplace_back(cl::sycl::device(cl::sycl::host_selector()));
  for (auto const& device: devices) {
    auto const& profile        = device.get_info<cl::sycl::info::device::profile>();
    auto const& driver_version = device.get_info<cl::sycl::info::device::driver_version>();
    auto const& version        = device.get_info<cl::sycl::info::device::version>();
    auto const& name           = device.get_info<cl::sycl::info::device::name>();
    auto const& vendor         = device.get_info<cl::sycl::info::device::vendor>();
    auto const& extensions     = device.get_info<cl::sycl::info::device::extensions>();
    log << "SYCL device:       " << name << '\n'
      << "  vendor:          " << vendor << '\n'
      << "  driver version:  " << driver_version << '\n'
      << "  SYCL version:    " << version << '\n'
      << "  profile:         " << profile << '\n'
      << "  extensions:     ";
    for (auto const& extension: extensions)
      log << " " << extension;
    log << '\n' << std::endl;
  }
}

#endif  // sycl_common_h
