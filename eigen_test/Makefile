.PHONY: all clean

TARGETS := test-cuda test-host

CXX         := /usr/bin/g++-8
CXXFLAGS    := -std=c++14 -O2 -g -pthread

# Eigen configuration
EIGEN_BASE := /data/cmssw/slc7_amd64_gcc820/external/eigen/e4c107b451c52c9ab2d7b7fa4194ee35332916ec-pafccj
EIGEN_FLAGS := -I$(EIGEN_BASE)/include/eigen3 -DEIGEN_DONT_PARALLELIZE

# CUDA configuration
CUDA_BASE   := /usr/local/cuda-10.1
CUDA_FLAGS  := -x cu -std=c++14 -O2 -g -w --expt-relaxed-constexpr --compiler-options "-pthread" -D FOR_CUDA
CUDA_CXX    := $(CUDA_BASE)/bin/nvcc

# Alpaka/Cupla configuration
ALPAKA_BASE := /opt/alpaka/alpaka
CUPLA_BASE  := /opt/alpaka/cupla
CUPLA_FLAGS := -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include 

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

# build Cuda Cupla code
test-cuda: testEigen.cc
	$(CUDA_CXX) $(CUDA_FLAGS) $(CUPLA_FLAGS) $(EIGEN_FLAGS) $^ -o $@

# build Cupla Host code
test-host: testEigen.cc
	$(CXX) $(CXXFLAGS) $(CUPLA_FLAGS) $(EIGEN_FLAGS) $^ -o $@
