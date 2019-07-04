#ifndef CMS_CUPLA_H
#define CMS_CUPLA_H

#ifdef FOR_CUDA
#include <cupla/standalone/GpuCudaRt.hpp>
#else
#include <cupla/standalone/CpuSerial.hpp>
#endif
#include <cuda_to_cupla.hpp>

#endif
