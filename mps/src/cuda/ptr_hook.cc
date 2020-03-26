#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <cstdint>
#include "cuda/ipc.hpp"

extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    __typeof__(cudaLaunchKernel) *launch_kernel_fp = (__typeof__(cudaLaunchKernel) *)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    // char *fun_cname = *(char **)((uintptr_t)func);
    std::cout << "Begin of maps " << getExcutionStart() << std::endl;
    std::cout << "Current ptr " << (std::uintptr_t)func << std::endl;
    cudaError_t ret = launch_kernel_fp(func, gridDim, blockDim, args, sharedMem, stream);
    return ret;
}