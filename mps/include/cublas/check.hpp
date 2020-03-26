/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/include/check.hpp
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Tuesday, December 18th 2018, 5:01:13 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 11:26:31 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once
#include <glog/logging.h>
#include <iostream>
// #include <boost/functional/hash.hpp>

#define CUDA_CHECK(condition)                                             \
    /* Code block avoids redefinition of cudaError_t error */             \
    do                                                                    \
    {                                                                     \
        cudaError_t error = condition;                                    \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS); \
  } while (0)// template <typename Container>
// struct container_hash
// {
//     std::size_t operator()(Container const &c) const
//     {
//         return boost::hash_range(c.begin(), c.end());
//     }
// };

enum CUBLAS_TYPE
{
    SGEMM = 0,
};