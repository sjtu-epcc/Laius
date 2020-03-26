/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/include/cudnn/compute/addtensor.hpp
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Monday, December 10th 2018, 7:47:33 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Monday, January 14th 2019, 9:52:13 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once
#include "cudnn/cudnn_api/base_compute.hpp"

class AddTensor : BaseCompute
{
  public:
    void setup(ShmCompute *shm_compute,
               DescriptorStore *descriptor_store,
               DeviceStore *device_store)
    {
        base_setup(shm_compute, descriptor_store, device_store);
    }

    void compute(cudnnHandle_t *cudnn_handle)
    {
        CUDNN_CHECK(cudnnAddTensor(*cudnn_handle,
                                   &alpha, tensor_in, tensor_in_ptr,
                                   &beta, tensor_out, tensor_out_ptr));
    }
};