/*
 * File: /Users/gema/Projects/Laius/code/mps/test/cublas_hook.cc
 * Project: /Users/gema/Projects/Laius/code/mps
 * Created Date: Monday, January 28th 2019, 7:40:10 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, February 1st 2019, 12:01:01 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2019 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
// #include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <iostream>
#include <unordered_map>
#include <pthread.h>
#include <semaphore.h>
CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const float *alpha, /* host or device pointer */
                                                     const float *A,
                                                     int lda,
                                                     const float *B,
                                                     int ldb,
                                                     const float *beta, /* host or device pointer */
                                                     float *C,
                                                     int ldc)
{
    std::cout << "@cublasSgemm hooked" << std::endl;
    std::cout << m << " " << n << " " << k << " " << lda << " " << ldb << " " << ldc << std::endl;
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);
    __typeof__(cublasSgemm_v2) *fp = (__typeof__(cublasSgemm_v2) *)dlsym(RTLD_NEXT, "cublasSgemm_v2");
    cublasStatus_t ret = fp(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float costtime;
    cudaEventElapsedTime(&costtime, start, stop);
    std::cout << costtime * 1000 << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ret;
}