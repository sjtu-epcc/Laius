/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/compute/activation_fwd.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Monday, December 10th 2018, 7:48:20 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, January 4th 2019, 11:16:44 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once

#include "optimized/cudnn_api/base_compute.hpp"

class ActivationFwd : BaseCompute
{
  public:
    std::unordered_map<uintptr_t, cudnnActivationDescriptor_t>::iterator activation_it;
    cudnnActivationDescriptor_t activation_desc;
    void setup(ShmCompute *shm_compute,
               DescriptorStore *descriptor_store,
               DeviceStore *device_store)
    {
        base_setup(shm_compute, descriptor_store, device_store);
        // activation_it = descriptor_store->activation_store.find(shm_compute->cudnn_call.activation_fwd_info.activation_name);
        // if (activation_it == descriptor_store->activation_store.end())
        // {
        //     DLOG(INFO) << __FILE__ << "@line" << __LINE__;
        //     std::abort();
        // };
        // activation_desc = activation_it->second;
        activation_desc = descriptor_store->activation_store[shm_compute->cudnn_call.activation_fwd_info.activation_name];
        // DLOG(INFO) << "input " << (uintptr_t)tensor_in << " input ptr " << (uintptr_t)tensor_in_ptr
        //    << "output " << (uintptr_t)tensor_out << " output ptr " << (uintptr_t)tensor_out_ptr
        //    << "activation desc" << (uintptr_t)activation_desc;
    }
    void compute(cudnnHandle_t *cudnn_handle)
    {
        CUDNN_CHECK(cudnnActivationForward(*cudnn_handle,
                                           activation_desc,
                                           &alpha, tensor_in, tensor_in_ptr,
                                           &beta, tensor_out, tensor_out_ptr));
    }
};