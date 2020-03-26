#pragma once
#include <iostream>
#include <pthread.h>
#include <glog/logging.h>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include "check.hpp"
#include "naive/cudnn_ipc.hpp"

class base_api
{
    //   private:
  public:
    float alpha, beta;
    cudnnTensorDescriptor_t tensor_in;
    cudnnTensorDescriptor_t tensor_out;
    void *tensor_in_ptr;
    void *tensor_out_ptr;

    //   public:
    base_api()
    {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_in));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_out));
    }
    void base_setup(SharedMemoryContents *shared_memory)
    {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(tensor_in,
                                                 cudnnDataType_t(shared_memory->cudnn_call.tensor_in.dataType),
                                                 shared_memory->cudnn_call.tensor_in.n,
                                                 shared_memory->cudnn_call.tensor_in.c,
                                                 shared_memory->cudnn_call.tensor_in.h,
                                                 shared_memory->cudnn_call.tensor_in.w,
                                                 shared_memory->cudnn_call.tensor_in.nStride,
                                                 shared_memory->cudnn_call.tensor_in.cStride,
                                                 shared_memory->cudnn_call.tensor_in.hStride,
                                                 shared_memory->cudnn_call.tensor_in.wStride));
        // std::cout << shared_memory->cudnn_call.tensor_in.device_handle.reserved << std::endl;
        // std::cout << shared_memory->cudnn_call.tensor_out.device_handle.reserved << std::endl;
        CUDA_CHECK(cudaIpcOpenMemHandle(&tensor_in_ptr,
                                        shared_memory->cudnn_call.tensor_in.device_handle,
                                        cudaIpcMemLazyEnablePeerAccess));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(tensor_out,
                                                 cudnnDataType_t(shared_memory->cudnn_call.tensor_out.dataType),
                                                 shared_memory->cudnn_call.tensor_out.n,
                                                 shared_memory->cudnn_call.tensor_out.c,
                                                 shared_memory->cudnn_call.tensor_out.h,
                                                 shared_memory->cudnn_call.tensor_out.w,
                                                 shared_memory->cudnn_call.tensor_out.nStride,
                                                 shared_memory->cudnn_call.tensor_out.cStride,
                                                 shared_memory->cudnn_call.tensor_out.hStride,
                                                 shared_memory->cudnn_call.tensor_out.wStride));
        // std::cout << tensor_out_ptr << std::endl;
        // std::cout << shared_memory->cudnn_call.tensor_out.device_handle.reserved << std::endl;
        CUDA_CHECK(cudaIpcOpenMemHandle(&tensor_out_ptr,
                                        shared_memory->cudnn_call.tensor_out.device_handle,
                                        cudaIpcMemLazyEnablePeerAccess));
        // std::cout << tensor_out_ptr << std::endl;
        alpha = shared_memory->cudnn_call.alpha;
        beta = shared_memory->cudnn_call.beta;
    }
    // void synchronize(SharedMemoryContents *shared_memory)
    // {
        // shared_memory->change_flag = 0;
        // CUDA_CHECK(cudaDeviceSynchronize());
    // }
    virtual void setup(SharedMemoryContents *shared_memory) = 0;
    virtual void compute(cudnnHandle_t &cudnn_handle, cudaEvent_t & event_,SharedMemoryContents *shared_memory) = 0;
    ~base_api()
    {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_in));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_out));
    }
};