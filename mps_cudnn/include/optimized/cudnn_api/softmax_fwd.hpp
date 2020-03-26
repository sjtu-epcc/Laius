/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/compute/softmax_fwd.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Monday, December 10th 2018, 7:48:52 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, January 4th 2019, 11:17:37 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once

#include "optimized/cudnn_api/base_compute.hpp"

class SoftmaxFwd : BaseCompute
{
  public:
    cudnnSoftmaxAlgorithm_t algo;
    cudnnSoftmaxMode_t mode;
    void setup(ShmCompute *shm_compute,
               DescriptorStore *descriptor_store,
               DeviceStore *device_store)
    {
        base_setup(shm_compute, descriptor_store, device_store);
        algo = shm_compute->cudnn_call.softmax_fwd_info.algo;
        mode = shm_compute->cudnn_call.softmax_fwd_info.mode;
    }

    void compute(cudnnHandle_t *cudnn_handle)
    {
        CUDNN_CHECK(cudnnSoftmaxForward(*cudnn_handle,
                                        algo, mode,
                                        &alpha, tensor_in, tensor_in_ptr,
                                        &beta, tensor_out, tensor_out_ptr));
    }
};