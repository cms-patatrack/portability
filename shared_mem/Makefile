.PHONY: all clean

TARGETS := test-cuda test-cupla-ptx test-cupla-host test-cupla-tbb test-sycl-spir test-sycl-ptx test-sycl-host

CXX         := /usr/bin/g++-8
CXXFLAGS    := -std=c++14 -O2 -g

# CUDA configuration
CUDA_BASE   := /usr/local/cuda-10.1
CUDA_FLAGS  := -x cu -std=c++14 -O2 -g -w --expt-relaxed-constexpr
CUDA_CXX    := $(CUDA_BASE)/bin/nvcc

# SYCL configuration
SYCL_BASE   := /opt/ComputeCpp
SYCL_CXX    := $(SYCL_BASE)/bin/compute++
SYCL_INFO   := $(SYCL_BASE)/bin/computecpp_info
SYCL_FLAGS  := -sycl-driver -no-serial-memop -O2 -std=c++17 -g -I$(SYCL_BASE)/include -L$(SYCL_BASE)/lib -lComputeCpp
SYCL_FLAGS  := $(SYCL_FLAGS) $(shell { $(SYCL_INFO) --dump-version | awk '{ print $$2 }'; echo '1.1.3 -DSYCL_WORKAROUND_WORK_ITEM_LOOP_BUG'; } | sort -h -r | head -n1 | awk '{ print $$2 }')
SYCL_CUDA   := -sycl-target ptx64 -DSYCL_TARGET_PTX --cuda-path=$(CUDA_BASE)
SYCL_SPIR   := -sycl-target spir64 -DSYCL_TARGET_SPIR
SYCL_HOST   := -DSYCL_TARGET_HOST

# Alpaka/Cupla configuration
ALPAKA_BASE := /opt/alpaka/alpaka
CUPLA_BASE  := /opt/alpaka/cupla
CUPLA_FLAGS := -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include -pthread
CUPLA_CUDA_ASYNC     := -DCUPLA_STREAM_ASYNC_ENABLED=1 -include "cupla/standalone/GpuCudaRt.hpp"
CUPLA_HOST_SYNC      := -DCUPLA_STREAM_ASYNC_ENABLED=0 -include "cupla/standalone/CpuSerial.hpp"
CUPLA_HOST_TBB_ASYNC := -DCUPLA_STREAM_ASYNC_ENABLED=1 -include "cupla/standalone/CpuTbbBlocks.hpp" -ltbb

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

# build CUDA code
test-cuda: cuda.cu
	$(CUDA_CXX) $(CUDA_FLAGS) $^ -o $@

# build Cupla code for the CUDA/PTX backend
test-cupla-ptx: cupla_v3.cc
	$(CUDA_CXX) $(CUDA_FLAGS) $(CUPLA_FLAGS) $(CUPLA_CUDA_ASYNC) $^ -o $@

# build Cupla code for the synchronous host backend
test-cupla-host: cupla_v3.cc
	$(CXX) $(CXXFLAGS) $(CUPLA_FLAGS) $(CUPLA_HOST_SYNC) $^ -o $@

# build Cupla code for the synchronous host backend
test-cupla-tbb: cupla_v3.cc
	$(CXX) $^ -o $@ $(CXXFLAGS) $(CUPLA_FLAGS) $(CUPLA_HOST_TBB_ASYNC)

# build SYCL_CXX code for the SPIR backend
test-sycl-spir: sycl.cc
	$(SYCL_CXX) $(SYCL_FLAGS) $(SYCL_SPIR) $^ -o $@

# build SYCL_CXX code for the CUDA/PTX backend
test-sycl-ptx: sycl.cc
	$(SYCL_CXX) $(SYCL_FLAGS) $(SYCL_CUDA) $^ -o $@

# build SYCL_CXX code for the host driver
test-sycl-host: sycl.cc
	$(SYCL_CXX) $(SYCL_FLAGS) $(SYCL_HOST) $^ -o $@

