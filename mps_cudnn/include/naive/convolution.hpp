#pragma once
#include "naive/base_api.hpp"

class convolution : base_api
{
  public:
    cudnnFilterDescriptor_t tensor_filter;
    cudnnConvolutionDescriptor_t tensor_conv;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_fwd_size;
    void *workspace_ptr, *tensor_filter_ptr;
    convolution()
    {
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&tensor_filter));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&tensor_conv));
    }
    void setup(SharedMemoryContents *shared_memory)
    {
        // std::cout << "begin base_setup" << std::endl;
        base_setup(shared_memory);
        // std::cout << shared_memory->cudnn_call.filter_desc.dataType << " "
        //   << shared_memory->cudnn_call.filter_desc.filter_format << " "
        //   << shared_memory->cudnn_call.filter_desc.k << " "
        //   << shared_memory->cudnn_call.filter_desc.c << " "
        //   << shared_memory->cudnn_call.filter_desc.h << " "
        //   << shared_memory->cudnn_call.filter_desc.w << std::endl;
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(tensor_filter,
                                               shared_memory->cudnn_call.filter_desc.dataType,
                                               shared_memory->cudnn_call.filter_desc.filter_format,
                                               shared_memory->cudnn_call.filter_desc.k,
                                               shared_memory->cudnn_call.filter_desc.c,
                                               shared_memory->cudnn_call.filter_desc.h,
                                               shared_memory->cudnn_call.filter_desc.w));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(tensor_conv,
                                                    shared_memory->cudnn_call.conv_desc.pad_h,
                                                    shared_memory->cudnn_call.conv_desc.pad_w,
                                                    shared_memory->cudnn_call.conv_desc.u,
                                                    shared_memory->cudnn_call.conv_desc.v,
                                                    shared_memory->cudnn_call.conv_desc.dilation_h,
                                                    shared_memory->cudnn_call.conv_desc.dilation_w,
                                                    shared_memory->cudnn_call.conv_desc.mode,
                                                    shared_memory->cudnn_call.conv_desc.computeType));
        algo = shared_memory->cudnn_call.conv_desc.algo;
        workspace_fwd_size = shared_memory->cudnn_call.conv_desc.workspace_size;
        // CUDA_CHECK(cudaIpcOpenMemHandle(&tensor_in_ptr,
        // shared_memory->cudnn_call.tensor_in.device_handle,
        // cudaIpcMemLazyEnablePeerAccess));
        CUDA_CHECK(cudaIpcOpenMemHandle(&tensor_filter_ptr,
                                        shared_memory->cudnn_call.filter_desc.device_handle,
                                        cudaIpcMemLazyEnablePeerAccess));
        // CUDA_CHECK(cudaIpcOpenMemHandle(&tensor_out_ptr,
        // shared_memory->cudnn_call.tensor_out.device_handle,
        // cudaIpcMemLazyEnablePeerAccess));
        if (workspace_fwd_size != 0)
            CUDA_CHECK(cudaIpcOpenMemHandle(&workspace_ptr,
                                            shared_memory->cudnn_call.conv_desc.workspace_handle,
                                            cudaIpcMemLazyEnablePeerAccess));
        else
            workspace_ptr = 0;
    }
    void compute(cudnnHandle_t &cudnn_handle, cudaEvent_t &event_, SharedMemoryContents *shared_memory)
    {
        CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle, &alpha,
                                            tensor_in, tensor_in_ptr,
                                            tensor_filter, tensor_filter_ptr,
                                            tensor_conv, algo,
                                            workspace_ptr, workspace_fwd_size,
                                            &beta, tensor_out, tensor_out_ptr));
        // if (shared_memory->change_flag == 1)
        // {
        // synchronize(shared_memory);
        // }
        shared_memory->process_flag = -1;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_in_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_out_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_filter_ptr));
        if (workspace_fwd_size != 0)
            CUDA_CHECK(cudaIpcCloseMemHandle(workspace_ptr));
        CUDA_CHECK(cudaEventRecord(event_));
        // CUDA_CHECK(cudaDeviceSynchronize());
        pthread_barrier_wait(&shared_memory->sync.barrier);
    }
};