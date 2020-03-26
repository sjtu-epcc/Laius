/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/optimized/cuda.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Friday, December 21st 2018, 3:31:21 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, January 4th 2019, 11:20:07 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once
#include "optimized/cudnn_api/activation_fwd.hpp"
#include "optimized/cudnn_api/addtensor.hpp"
#include "optimized/cudnn_api/conv_fwd.hpp"
#include "optimized/cudnn_api/pooling_fwd.hpp"
#include "optimized/cudnn_api/softmax_fwd.hpp"
#include "optimized/cudnn_ipc.hpp"
// #include "optimized/cuda_ipc.hpp"

struct ComputeArgs
{
    ShmCompute *shm_compute;
    int *cur_percent_;
    int *cur_pid_;
    // cudaEvent_t *start_;
    // cudaEvent_t *complete_;
    cudnnHandle_t *cudnn;
    DeviceStore *device_store;
    DescriptorStore *descriptor_store;
};

void *compute_server(void *compute_args_);
void *compute_server_test(void *compute_args_);

struct SetArgs
{
    ShmSet *shm_set;
    DescriptorStore *descriptor_store;
};

void *set_server(void *set_args_);