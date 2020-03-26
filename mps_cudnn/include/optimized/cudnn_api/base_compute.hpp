/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/compute/base_compute.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Tuesday, December 11th 2018, 3:40:19 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, January 4th 2019, 11:16:25 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once

#include <unordered_map>
#include <cudnn.h>
#include "optimized/cudnn_ipc.hpp"

class BaseCompute
{
  public:
    float alpha, beta;
    std::unordered_map<uintptr_t, cudnnTensorDescriptor_t>::iterator tensor_it;
    cudnnTensorDescriptor_t tensor_in;
    cudnnTensorDescriptor_t tensor_out;
    std::unordered_map<uintptr_t, MallocInfo>::iterator malloc_it;
    MallocInfo tmp_malloc;
    void *tensor_in_ptr;
    void *tensor_out_ptr;

    void base_setup(ShmCompute *shm_compute, DescriptorStore *descriptor_store, DeviceStore *device_store)
    {
        // pthread_mutex_lock(&descriptor_store->mutex);
        // tensor_it = descriptor_store->tensor_store.find(shm_compute->cudnn_call.tensor_in_info.tensor_name);
        // if (tensor_it == descriptor_store->tensor_store.end())
        // {
        //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
        //     std::abort();
        // };
        // tensor_in = tensor_it->second;
        tensor_in = descriptor_store->tensor_store[shm_compute->cudnn_call.tensor_in_info.tensor_name];
        // tensor_it = descriptor_store->tensor_store.find(shm_compute->cudnn_call.tensor_out_info.tensor_name);
        // if (tensor_it == descriptor_store->tensor_store.end())
        // {
        //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
        //     std::abort();
        // };
        // tensor_out = tensor_it->second;
        tensor_out = descriptor_store->tensor_store[shm_compute->cudnn_call.tensor_out_info.tensor_name];
        if (shm_compute->cudnn_call.tensor_in_info.if_tensor_handle)
        {
            CUDA_CHECK(cudaIpcOpenMemHandle(&tmp_malloc.device_ptr,
                                            shm_compute->cudnn_call.tensor_in_info.tensor_handle,
                                            cudaIpcMemLazyEnablePeerAccess));
            tensor_in_ptr = tmp_malloc.device_ptr;
            pthread_mutex_lock(&device_store->mutex);
            device_store->malloc_store.emplace(shm_compute->cudnn_call.tensor_in_info.tensor_ptr,
                                               tmp_malloc);
            pthread_mutex_unlock(&device_store->mutex);
        }
        else
        {
            tensor_in_ptr = device_store->malloc_store[shm_compute->cudnn_call.tensor_in_info.tensor_ptr].device_ptr;
            // if (tensor_in_ptr == nullptr)
            // {
            //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
            //     std::abort();
            // }
        }
        if (shm_compute->cudnn_call.tensor_out_info.if_tensor_handle)
        {
            CUDA_CHECK(cudaIpcOpenMemHandle(&tmp_malloc.device_ptr,
                                            shm_compute->cudnn_call.tensor_out_info.tensor_handle,
                                            cudaIpcMemLazyEnablePeerAccess));
            tensor_out_ptr = tmp_malloc.device_ptr;
            pthread_mutex_lock(&device_store->mutex);
            device_store->malloc_store.emplace(shm_compute->cudnn_call.tensor_out_info.tensor_ptr,
                                               tmp_malloc);
            pthread_mutex_unlock(&device_store->mutex);
        }
        else
        {
            tensor_out_ptr = device_store->malloc_store[shm_compute->cudnn_call.tensor_out_info.tensor_ptr].device_ptr;
            // if (tensor_out_ptr == nullptr)
            // {
            //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
            //     std::abort();
            // }
        }
        alpha = shm_compute->cudnn_call.alpha;
        beta = shm_compute->cudnn_call.beta;
    }
    virtual void setup(ShmCompute *shm_compute,
                       DescriptorStore *descriptor_store,
                       DeviceStore *device_store) = 0;
    virtual void compute(cudnnHandle_t *cudnn_handle) = 0;
};