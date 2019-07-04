.PHONY: all clean

TARGETS := test-cuda test-host

CXX         := g++
CXXFLAGS    := -std=c++14 -O2 -g -pthread

# Eigen configuration
EIGEN_BASE := /data/cmssw/slc7_amd64_gcc820/external/eigen/e4c107b451c52c9ab2d7b7fa4194ee35332916ec-pafccj
EIGEN_FLAGS := -I$(EIGEN_BASE)/include/eigen3 -DEIGEN_DONT_PARALLELIZE

# CUDA configuration
CUDA_BASE   := /data/cmssw/slc7_amd64_gcc820/external/cuda/10.1.105-pafccj2
CUDA_FLAGS  := -x cu -std=c++14 -O2 -g -w --expt-relaxed-constexpr --compiler-options "-pthread"
CUDA_CXX    := $(CUDA_BASE)/bin/nvcc

# Alpaka/Cupla configuration
ALPAKA_BASE := /data/user/dmarcato/alpaka
CUPLA_BASE  := /data/user/dmarcato/cupla
CUPLA_FLAGS := -I$(ALPAKA_BASE)/include -I$(CUPLA_BASE)/include 

# Boost configuration
BOOST_BASE := /data/cmssw/slc7_amd64_gcc700/external/boost/1.67.0-pafccj
BOOST_FLAGS := -I$(BOOST_BASE)/include

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

# build Cuda Cupla code
test-cuda: testEigen.cc
	$(CUDA_CXX) $(CUDA_FLAGS) $(CUPLA_FLAGS) $(EIGEN_FLAGS) $(BOOST_FLAGS) $^ -o $@

# build Cupla Host code
test-host: testEigen.cc
	$(CXX) $(CXXFLAGS) $(CUPLA_FLAGS) $(EIGEN_FLAGS) $(BOOST_FLAGS) $^ -o $@
