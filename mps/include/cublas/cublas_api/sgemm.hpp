/*
 * File: /Users/gema/Projects/Laius/code/mps/test/cublas/cublas_api/sgemm.hpp
 * Project: /Users/gema/Projects/Laius/code/mps
 * Created Date: Tuesday, January 29th 2019, 2:25:16 am
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 29th 2019, 11:28:03 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#pragma once
#include "cublas/cublas_api/base_api.hpp"

class Sgemm : BaseApi
{
  public:
    cublasOperation_t transa;
    cublasOperation_t transb;
    float alpha, beta;
    float *A, *B, *C;
    int m, n, k, lda, ldb, ldc;
    void set_up(ShmCublas *shm_cublas, DeviceStore *dev_store)
    {
        transa = shm_cublas->cublas_call.sgemm_info.transa;
        transb = shm_cublas->cublas_call.sgemm_info.transb;
        alpha = shm_cublas->cublas_call.sgemm_info.alpha;
        beta = shm_cublas->cublas_call.sgemm_info.beta;
        m = shm_cublas->cublas_call.sgemm_info.m;
        n = shm_cublas->cublas_call.sgemm_info.n;
        k = shm_cublas->cublas_call.sgemm_info.k;
        lda = shm_cublas->cublas_call.sgemm_info.lda;
        ldb = shm_cublas->cublas_call.sgemm_info.ldb;
        ldc = shm_cublas->cublas_call.sgemm_info.ldc;
        A = get_devptr<float>(shm_cublas->cublas_call.sgemm_info.A_devptr,
                              shm_cublas->cublas_call.sgemm_info.A_handle,
                              dev_store);
        B = get_devptr<float>(shm_cublas->cublas_call.sgemm_info.B_devptr,
                              shm_cublas->cublas_call.sgemm_info.B_handle,
                              dev_store);
        C = get_devptr<float>(shm_cublas->cublas_call.sgemm_info.C_devptr,
                              shm_cublas->cublas_call.sgemm_info.C_handle,
                              dev_store);
    }
    void compute(cublasHandle_t *cublas_handle)
    {
        CUBLAS_CHECK(cublasSgemm_v2(*cublas_handle,
                                    transa,
                                    transb,
                                    m, n, k,
                                    &alpha, A, lda,
                                    B, ldb, &beta,
                                    C, ldc));
        DLOG(INFO) << "------cublas-sgemm-completed------";
    }
};