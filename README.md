# A comparison of different heterogeneous frameworks

The purpose of these test programs is to experiment with various heterogeneous frameworks and "performance portability" approaches.

## Current implementations

| Test          | CPU (serial)  | CUDA  | Alpaka/Cupla  | SYCL  |
|---------------|---------------|-------|---------------|-------|
| `vector_add`  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: :heavy_check_mark: :heavy_check_mark: :heavy_check_mark: | :heavy_check_mark: :heavy_check_mark: :heavy_check_mark: |
| `shared_mem`  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: :heavy_check_mark: :heavy_check_mark: :heavy_check_mark: | :heavy_check_mark: :heavy_check_mark: :heavy_check_mark: |
| `eigen_test`  | :x: | :x: | :heavy_check_mark: :x: :x: :heavy_check_mark: | :x: :x: :x: |

Support for **Alpaka/Cupla** is split in
  - CPU serial
  - CPU paraller, using OpenMP
  - CPU paraller, using TBB
  - GPU parallel, using CUDA

Support for **SYCL** is split in
  - CPU (unspecified)
  - genericl OpenCL
  - Nvidia PTX


### CUDA

These programs require a recent CUDA version (`nvcc` supporting C++14 and `--expt-relaxed-constexpr`) and a machine with GPU.


### Alpaka/Cupla

Alpaka/Cupla support different backends; so far it has been tested with serial, TBB and OpenMP backends for the CPU, and the CUDA
backend for Nvidia GPUs. The latter requires CUDA 9.2 through 10.1, and has been tested with gcc 7.x and gcc 8.x.

Rather than using the advertised `CMake`-based approach, one can use Cupla as header-only library:
```
# choose a base directory for Alpaka and Cupla
BASE=/opt/alpaka
mkdir -p $BASE
cd $BASE

git clone -b develop git@github.com:ComputationalRadiationPhysics/alpaka.git
git clone -b dev     git@github.com:cms-patatrack/cupla.git
```

The `cms-patatrack` for of Cupla includes
  - the latest development version of Cupla;
  - support for the latest development version of Alpaka ;
  - merge of [cupla#112](https://github.com/ComputationalRadiationPhysics/cupla/pull/112) and following bug fixes;
  - removal of the embedded version of Alpaka.

Note: these instructions were accurate as of July 1st, 2019.

See the [`vector_add/Makefile`](vector_add/Makefile) for an example.
If you use a different `BASE` folder, updated the `Makefile`s accordingly.


### SYCL

These programs were tested with CodePlay's [ComputeCpp Community Edition](https://developer.codeplay.com/products/computecpp/ce/home/),
version 1.1.3 .


## Other projects

Other projects with similar goals:
  - BabelStream memory transfer benchmark: https://uob-hpc.github.io/BabelStream/
  - Parallel Research Tools: https://github.com/ParRes/Kernels
  - DoE portability study: https://performanceportability.org/
