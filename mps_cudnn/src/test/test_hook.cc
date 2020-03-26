/*
 * File: /home/haohao/Projects/Paper/reference/mps/src/test/test_hook.cc
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Sunday, December 23rd 2018, 7:08:38 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Monday, December 24th 2018, 1:56:15 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#include <dlfcn.h>
#include "optimized/cuda_ipc.hpp"
struct CudaFunctions {
  __typeof__(cudaMalloc) *Malloc;
  __typeof__(cudaFree) *Free;
//   __typeof__(cudaLaunchKernel) *LaunchKernel;
//   __typeof__(cudaMemcpyToSymbol) *MemcpyToSymbol;
//   __typeof__(cudaMemcpyToSymbolAsync) *MemcpyToSymbolAsync;
  CudaFunctions() {
    Malloc = (__typeof__(cudaMalloc)*)dlsym(RTLD_NEXT, "cudaMalloc");
    Free = (__typeof__(cudaFree)*)dlsym(RTLD_NEXT, "cudaFree");
    // LaunchKernel = (__typeof__(cudaLaunchKernel)*)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    // MemcpyToSymbol = (__typeof__(cudaMemcpyToSymbol)*)dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");
    // MemcpyToSymbolAsync = (__typeof__(cudaMemcpyToSymbolAsync)*)dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync");
  }
};

CudaFunctions cuda_func;
static MallocInterProcess malloc_inter_process;
static pthread_mutex_t api_used = PTHREAD_MUTEX_INITIALIZER;
MallocInfo malloc_info;

extern "C"
{
    __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
    {
        std::cout << "Malloc hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        cudaError_t malloc_ret = cuda_func.Malloc(devPtr, size);
        assert(malloc_ret == cudaSuccess);
        // CHECK_EQ(malloc_ret, cudaSuccess) << "Malloc Failed";
        malloc_info.size = size;
        malloc_info.device_ptr = (uintptr_t)*devPtr;
        CUDA_CHECK(cudaIpcGetMemHandle(&malloc_info.device_handle, *devPtr));
        std::cout << malloc_info.device_handle.reserved << " "
                  << malloc_info.device_ptr << " "
                  << malloc_info.size << std::endl;
        // CHECK_EQ(get_ret, cudaSuccess) << "IpcMemHandle get failed";
        malloc_inter_process.send_malloc(malloc_info);
        pthread_mutex_unlock(&api_used);
        return malloc_ret;
    }
}