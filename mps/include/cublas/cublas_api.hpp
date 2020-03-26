/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/include/cudnn/cuda.hpp
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Friday, December 21st 2018, 3:31:21 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 12:08:36 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */

#pragma once
#include "cublas/cublas_api/base_api.hpp"
#include "cublas/cublas_api/sgemm.hpp"

void cublas_api_server(cublasHandle_t *cublas_handle,
                ShmCublas *shm_cublas,
                DeviceStore *dev_store,
                int &cur_percent_, int &cur_pid_);