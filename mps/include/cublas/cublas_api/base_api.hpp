/*
 * File: /Users/gema/Projects/Laius/code/mps/test/cublas/cublas_api/base_api.hpp
 * Project: /Users/gema/Projects/Laius/code/mps
 * Created Date: Tuesday, January 29th 2019, 2:25:30 am
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 1:57:06 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once
#include <cublas_v2.h>
#include <unordered_map>
#include <cublas/cublas_ipc.hpp>

class BaseApi
{
  public:
    std::unordered_map<std::uintptr_t, void *>::iterator malloc_it;
    virtual void set_up(ShmCublas *shm_cublas, DeviceStore *dev_store) = 0;
    virtual void compute(cublasHandle_t *cublas_handle) = 0;
    template <typename T>
    T *get_devptr(std::uintptr_t &logic_ptr,
                  cudaIpcMemHandle_t &logic_memhandle,
                  DeviceStore *dev_store)
    {
        T *ret;
        malloc_it = dev_store->malloc_store.find(logic_ptr);
        if (malloc_it != dev_store->malloc_store.end())
        {
            DLOG(INFO) << "------already-opend-fptr------";
            ret = (T *)malloc_it->second;
        }
        else
        {
            CUDA_CHECK(cudaIpcOpenMemHandle((void **)&ret,
                                            logic_memhandle,
                                            cudaIpcMemLazyEnablePeerAccess));
            dev_store->malloc_store.emplace(logic_ptr, (void *)ret);
            DLOG(INFO) << "------fptr-opend------";
        }
        return ret;
    }
};