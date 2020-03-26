/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/compute/pooling_fwd.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Monday, December 10th 2018, 7:48:38 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, January 4th 2019, 11:17:18 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once

#include "optimized/cudnn_api/base_compute.hpp"

class PoolingFwd : BaseCompute
{
  public:
    std::unordered_map<uintptr_t, cudnnPoolingDescriptor_t>::iterator pooling_it;
    cudnnPoolingDescriptor_t pooling_desc;

    void setup(ShmCompute *shm_compute,
               DescriptorStore *descriptor_store,
               DeviceStore *device_store)
    {
        base_setup(shm_compute, descriptor_store, device_store);
        // pooling_it = descriptor_store->pooling_store.find(shm_compute->cudnn_call.pooling_fwd_info.pooling_name);
        // if (pooling_it == descriptor_store->pooling_store.end())
        // {
        //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
        //     std::abort();
        // };
        // pooling_desc = pooling_it->second;
        pooling_desc = descriptor_store->pooling_store[shm_compute->cudnn_call.pooling_fwd_info.pooling_name];
    }

    void compute(cudnnHandle_t *cudnn_handle)
    {
        CUDNN_CHECK(cudnnPoolingForward(*cudnn_handle,
                                        pooling_desc,
                                        &alpha, tensor_in, tensor_in_ptr,
                                        &beta, tensor_out, tensor_out_ptr));
    }
};