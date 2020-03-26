#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <pthread.h>
#include "cuda/ipc.hpp"

struct CudaFunctions
{
    __typeof__(cudaMalloc) *Malloc;
    __typeof__(cudaFree) *Free;
    __typeof__(cudaLaunchKernel) *LaunchKernel;
    __typeof__(cudaMemcpyToSymbol) *MemcpyToSymbol;
    __typeof__(cudaMemcpyToSymbolAsync) *MemcpyToSymbolAsync;
    CudaFunctions()
    {
        Malloc = (__typeof__(cudaMalloc) *)dlsym(RTLD_NEXT, "cudaMalloc");
        Free = (__typeof__(cudaFree) *)dlsym(RTLD_NEXT, "cudaFree");
        LaunchKernel = (__typeof__(cudaLaunchKernel) *)dlsym(RTLD_NEXT, "cudaLaunchKernel");
        MemcpyToSymbol = (__typeof__(cudaMemcpyToSymbol) *)dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");
        MemcpyToSymbolAsync = (__typeof__(cudaMemcpyToSymbolAsync) *)dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync");
    }
};
CudaFunctions nv;
InterProcess ip(true);

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    printf("wrapped cudaMalloc\n");
    cudaError_t result = nv.Malloc(devPtr, size);
    ip.add_memory_allocation(*devPtr);
    return result;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    // pthread_mutex_lock(&rt_api);
    printf("wrapped cudaFree\n");
    // pthread_mutex_unlock(&rt_api);
    return nv.Free(devPtr);
}

__host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                                void **args, size_t sharedMem, cudaStream_t stream)
{
    // printf("cuda_wrap: cudaLaunchKernel\n");
    ip.send_kernel(func, gridDim, blockDim, args, sharedMem);
    return cudaSuccess;
}

// __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
//                                                   size_t offset, cudaMemcpyKind kind)
// {
//     printf("cuda_wrap: cudaMemcpyToSymbol\n");
//     return nv.MemcpyToSymbol(symbol, src, count, offset, kind);
// }

// __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count,
//                                                        size_t offset, cudaMemcpyKind kind, cudaStream_t stream)
// {
//     printf("cuda_wrap: cudaMemcpyToSymbolAsync\n");
//     return nv.MemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
// }
