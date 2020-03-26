/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/compute/conv_fwd.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Monday, December 10th 2018, 7:47:47 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, January 4th 2019, 11:16:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once

#include "optimized/cudnn_api/base_compute.hpp"

class ConvFwd : BaseCompute
{
  public:
    std::unordered_map<uintptr_t, cudnnConvolutionDescriptor_t>::iterator conv_it;
    cudnnConvolutionDescriptor_t conv_desc;
    std::unordered_map<uintptr_t, cudnnFilterDescriptor_t>::iterator filter_it;
    cudnnFilterDescriptor_t filter_desc;
    void *filter_desc_ptr;
    cudnnConvolutionFwdAlgo_t algo;
    void *workspace;
    size_t workspaceSizeInBytes;

    void setup(ShmCompute *shm_compute,
               DescriptorStore *descriptor_store,
               DeviceStore *device_store)
    {
        base_setup(shm_compute, descriptor_store, device_store);
        // conv_it = descriptor_store->conv_store.find(shm_compute->cudnn_call.conv_fwd_info.conv_name);
        // if (conv_it == descriptor_store->conv_store.end())
        // {
        //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
        //     std::abort();
        // };
        // conv_desc = conv_it->second;
        conv_desc = descriptor_store->conv_store[shm_compute->cudnn_call.conv_fwd_info.conv_name];
        // filter_it = descriptor_store->filter_store.find(shm_compute->cudnn_call.conv_fwd_info.filter_name);
        // if (filter_it == descriptor_store->filter_store.end())
        // {
        //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
        //     std::abort();
        // };
        // filter_desc = filter_it->second;
        filter_desc = descriptor_store->filter_store[shm_compute->cudnn_call.conv_fwd_info.filter_name];
        if (shm_compute->cudnn_call.conv_fwd_info.if_filter_handle)
        {
            CUDA_CHECK(cudaIpcOpenMemHandle(&tmp_malloc.device_ptr,
                                            shm_compute->cudnn_call.conv_fwd_info.filter_handle,
                                            cudaIpcMemLazyEnablePeerAccess));
            filter_desc_ptr = tmp_malloc.device_ptr;
            pthread_mutex_lock(&device_store->mutex);
            device_store->malloc_store.emplace(shm_compute->cudnn_call.conv_fwd_info.filter_ptr, tmp_malloc);
            pthread_mutex_unlock(&device_store->mutex);
        }
        else
        {
            filter_desc_ptr = device_store->malloc_store[shm_compute->cudnn_call.conv_fwd_info.filter_ptr].device_ptr;
            // if (filter_desc_ptr == nullptr)
            // {
            //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
            //     std::abort();
            // }
        }
        algo = shm_compute->cudnn_call.conv_fwd_info.algo;
        if (shm_compute->cudnn_call.conv_fwd_info.workspace_size != 0)
        {
            if (shm_compute->cudnn_call.conv_fwd_info.if_workspace_handle)
            {
                CUDA_CHECK(cudaIpcOpenMemHandle(&tmp_malloc.device_ptr,
                                                shm_compute->cudnn_call.conv_fwd_info.workspace_handle,
                                                cudaIpcMemLazyEnablePeerAccess));
                workspace = tmp_malloc.device_ptr;
                pthread_mutex_lock(&device_store->mutex);
                device_store->malloc_store.emplace(shm_compute->cudnn_call.conv_fwd_info.workspace_ptr, tmp_malloc);
                pthread_mutex_unlock(&device_store->mutex);
            }
            else
            {
                workspace = device_store->malloc_store[shm_compute->cudnn_call.conv_fwd_info.workspace_ptr].device_ptr;
                // if (workspace == nullptr)
                // {
                //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
                //     std::abort();
                // }
            }
        }
        else
            workspace = nullptr;
        workspaceSizeInBytes = shm_compute->cudnn_call.conv_fwd_info.workspace_size;
        // DLOG(INFO) << "alpha:" << alpha << " beta:" << beta
        //    << " x:" << (uintptr_t)tensor_in << " x_ptr:" << (uintptr_t)tensor_in_ptr;
        // DLOG(INFO) << "w:" << (uintptr_t)filter_desc << " w_ptr:" << (uintptr_t)filter_desc_ptr
        //    << " algo:" << algo << " workspace_ptr:" << (uintptr_t)workspace
        //    << " workSpaceSizeInBytes:" << workspaceSizeInBytes;
        // DLOG(INFO) << "y:" << (uintptr_t)tensor_out << " y_ptr:" << (uintptr_t)tensor_out_ptr;
    }

    void compute(cudnnHandle_t *cudnn_handle)
    {
        CUDNN_CHECK(cudnnConvolutionForward(*cudnn_handle,
                                            &alpha, tensor_in, tensor_in_ptr,
                                            filter_desc, filter_desc_ptr, conv_desc,
                                            algo, workspace, workspaceSizeInBytes,
                                            &beta, tensor_out, tensor_out_ptr));
    }
};