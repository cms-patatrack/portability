#ifndef cuda_common_h
#define cuda_common_h

#include <iomanip>
#include <utility>
#include <vector>

#include <cuda_runtime.h>


constexpr
unsigned int getCudaCoresPerSM(unsigned int major, unsigned int minor) {
  switch (major * 10 + minor) {
  // Fermi architecture
  case 20:  // SM 2.0: GF100 class
    return  32;
  case 21:  // SM 2.1: GF10x class
    return  48;

  // Kepler architecture
  case 30:  // SM 3.0: GK10x class
  case 32:  // SM 3.2: GK10x class
  case 35:  // SM 3.5: GK11x class
  case 37:  // SM 3.7: GK21x class
    return 192;

  // Maxwell architecture
  case 50:  // SM 5.0: GM10x class
  case 52:  // SM 5.2: GM20x class
  case 53:  // SM 5.3: GM20x class
    return 128;

  // Pascal architecture
  case 60:  // SM 6.0: GP100 class
    return  64;
  case 61:  // SM 6.1: GP10x class
  case 62:  // SM 6.2: GP10x class
    return 128;

  // Volta architecture
  case 70:  // SM 7.0: GV100 class
  case 72:  // SM 7.2: GV11b class
    return  64;

  // Turing architecture
  case 75:  // SM 7.5: TU10x class
    return  64;

  // unknown architecture, return a default value
  default:
    return  64;
  }
}


// list CUDA devices
template <typename T>
void list_devices(T & log) {
  int numberOfDevices = 0;
  std::vector<std::pair<int, int>> computeCapabilities;

  auto status = cudaGetDeviceCount(&numberOfDevices);
  if (cudaSuccess != status) {
    std::cerr << "Failed to initialize the CUDA runtime" << '\n';
    return;
  }
  computeCapabilities.reserve(numberOfDevices);
  log << "CUDA runtime successfully initialised, found " << numberOfDevices << " compute devices\n";

  for (int i = 0; i < numberOfDevices; ++i) {
    // read information about the compute device.
    // see the documentation of cudaGetDeviceProperties() for more information.
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);
    log << "\nCUDA device " << i << ": " << properties.name << '\n';

    // compute capabilities
    log << "  compute capability:          " << properties.major << "." << properties.minor << " (sm_" << properties.major << properties.minor << ")\n";
    computeCapabilities.emplace_back(properties.major, properties.minor);
    log << "  streaming multiprocessors: " << std::setw(13) << properties.multiProcessorCount << '\n';
    log << "  CUDA cores: " << std::setw(28) << properties.multiProcessorCount * getCudaCoresPerSM(properties.major, properties.minor ) << '\n';
    log << "  single to double performance: " << std::setw(8) << properties.singleToDoublePrecisionPerfRatio << ":1\n";

    // compute mode
    static constexpr const char* computeModeDescription[] = {
      "default (shared)",               // cudaComputeModeDefault
      "exclusive (single thread)",      // cudaComputeModeExclusive
      "prohibited",                     // cudaComputeModeProhibited
      "exclusive (single process)",     // cudaComputeModeExclusiveProcess
      "unknown"
    };
    log << "  compute mode:" << std::right << std::setw(27) << computeModeDescription[std::min<int>(properties.computeMode, sizeof(computeModeDescription)/sizeof(char*) - 1)] << '\n';

    // TODO if a device is in exclusive use, skip it and remove it from the list, instead of failing with abort()
    cudaSetDevice(i);
    cudaSetDeviceFlags(cudaDeviceScheduleAuto | cudaDeviceMapHost);

    // read the free and total amount of memory available for allocation by the device, in bytes.
    // see the documentation of cudaMemGetInfo() for more information.
    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    log << "  memory: " << std::setw(6) << freeMemory / (1 << 20) << " MB free / " << std::setw(6) << totalMemory / (1 << 20) << " MB total\n";
    log << "  constant memory:               " << std::setw(6) << properties.totalConstMem / (1 << 10) << " kB\n";
    log << "  L2 cache size:                 " << std::setw(6) << properties.l2CacheSize / (1 << 10) << " kB\n";

    // L1 cache behaviour
    static constexpr const char* l1CacheModeDescription[] = {
      "unknown",
      "local memory",
      "global memory",
      "local and global memory"
    };
    int l1CacheMode = properties.localL1CacheSupported + 2 * properties.globalL1CacheSupported;
    log << "  L1 cache mode:" << std::setw(26) << std::right << l1CacheModeDescription[l1CacheMode] << '\n';
    log << '\n';

    log << "Other capabilities\n";
    log << "  " << (properties.canMapHostMemory ? "can" : "cannot") << " map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer()\n";
    log << "  " << (properties.pageableMemoryAccess ? "supports" : "does not support") << " coherently accessing pageable memory without calling cudaHostRegister() on it\n";
    log << "  " << (properties.pageableMemoryAccessUsesHostPageTables ? "can" : "cannot") << " access pageable memory via the host's page tables\n";
    log << "  " << (properties.canUseHostPointerForRegisteredMem ? "can" : "cannot") << " access host registered memory at the same virtual address as the host\n";
    log << "  " << (properties.unifiedAddressing ? "shares" : "does not share") << " a unified address space with the host\n";
    log << "  " << (properties.managedMemory ? "supports" : "does not support") << " allocating managed memory on this system\n";
    log << "  " << (properties.concurrentManagedAccess ? "can" : "cannot") << " coherently access managed memory concurrently with the host\n";
    log << "  " << "the host " << (properties.directManagedMemAccessFromHost ? "can" : "cannot") << " directly access managed memory on the device without migration\n";
    log << "  " << (properties.cooperativeLaunch ? "supports" : "does not support") << " launching cooperative kernels via cudaLaunchCooperativeKernel()\n";
    log << "  " << (properties.cooperativeMultiDeviceLaunch ? "supports" : "does not support") << " launching cooperative kernels via cudaLaunchCooperativeKernelMultiDevice()\n";
    log << '\n';
    // set and read the CUDA resource limits.
    // see the documentation of cudaDeviceSetLimit() for more information.

    size_t value;
    log << "CUDA limits\n";
    cudaDeviceGetLimit(&value, cudaLimitPrintfFifoSize);
    log << "  printf buffer size:        " << std::setw(10) << value / (1 << 20) << " MB\n";
    cudaDeviceGetLimit(&value, cudaLimitStackSize);
    log << "  stack size:                " << std::setw(10) << value / (1 << 10) << " kB\n";
    cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize);
    log << "  malloc heap size:          " << std::setw(10) << value / (1 << 20) << " MB\n";
    if ((properties.major > 3) or (properties.major == 3 and properties.minor >= 5)) {
      cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth);
      log << "  runtime sync depth:           " << std::setw(10) << value << '\n';
      cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount);
      log << "  runtime pending launch count: " << std::setw(10) << value << '\n';
    }
  }
}

#endif  // cuda_common_h
