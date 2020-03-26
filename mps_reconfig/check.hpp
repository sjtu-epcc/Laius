/*
 * File: /home/haohao/Projects/Paper/reference/mps/include/check.hpp
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Tuesday, December 18th 2018, 5:01:13 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, December 18th 2018, 8:46:04 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once
#include <glog/logging.h>
#include <iostream>

#define CUDA_CHECK(condition)                                             \
    /* Code block avoids redefinition of cudaError_t error */             \
    do                                                                    \
    {                                                                     \
        cudaError_t error = condition;                                    \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUDNN_CHECK(expression)                                    \
    {                                                              \
        cudnnStatus_t status = (expression);                       \
        if (status != CUDNN_STATUS_SUCCESS)                        \
        {                                                          \
            std::cerr << "Error on line " << __LINE__ << ": "      \
                      << cudnnGetErrorString(status) << std::endl; \
            std::exit(EXIT_FAILURE);                               \
        }                                                          \
    }

enum API_TYPE
{
    ADDTENSOR_N = 0,
    CONVFWD_N = 1,
    ACTIVATIONFWD_N = 2,
    POOLFWD_N = 3,
    SOFTMAXFWD_N = 4,
};

enum COMPUTE_TYPE
{
    ADDTENSOR = 0,
    CONVFWD = 1,
    ACTIVATIONFWD = 2,
    POOLFWD = 3,
    SOFTMAXFWD = 4,
};

enum SET_TYPE
{
    SETTENSOR4D = 0,
    SETTENSOR4DEX = 1,
    SETCONV2D = 2,
    SETFILTER4D = 3,
    SETACTIVATION = 4,
    SETPOOLING2D = 5,
};