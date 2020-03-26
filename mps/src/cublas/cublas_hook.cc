/*
 * File: /Users/gema/Projects/Laius/code/mps/test/cublas_hook.cc
 * Project: /Users/gema/Projects/Laius/code/mps
 * Created Date: Monday, January 28th 2019, 7:40:10 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Friday, February 1st 2019, 1:55:03 pm
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
#include "cublas/cublas_ipc.hpp"
#include "cublas/check.hpp"
struct predict
{
    double duration[11];
    double bandwidth[11];
    void init(double d[], double b[])
    {
        for (int i = 3; i <= 10; i++)
        {
            duration[i] = d[i-3];
            bandwidth[i] = b[i-3];
        }
    }
};
static std::unordered_map<int, predict> prediction;
void init_prediction()
{
    predict sgem;
    int key_0 = 48484;
    int key_1 = 47886;
    int key_2 = 47974;
    int key_3 = 47121;
    int key_4 = 47968;
    int key_5 = 47112;
    int key_6 = 95393;
    int key_7 = 94570;
    int key_8 = 94602;
    int key_9 = 93721;
    
    double b_0[8] = {99.18,89.26,81.03,72.06,63.1,50.4,41.05,31.98};
    double b_1[8] = {445.5,445.465,437.61,424.065,402.02,351.29,298.4,238.265};
    double b_2[8] = {136.29,124.94,113.07,98.04,86.28,70.26,58.26,45.94};
    double b_3[8] = {94.05,88.77,81.49,72.55,64.95,54.11,45.55,36.61};
    double b_4[8] = {135.65,123.87,112.3,97.32,85.7,69.38,57.82,45.51};
    double b_5[8] = {86.14,80.59,74.64,66.17,59.54,49.57,41.85,33.7};
    double b_6[8] = {92.91,84.64,76.34,68.84,59.41,47.43,38.66,30.01};
    double b_7[8] = {452.4,452.55,442.725,432.935,412.465,361,305.98,243};
    double b_8[8] = {228.92,221.2,212.73,181.55,160.57,130.51,108.04,85.14};
    double b_9[8] = {201.23,193.52,187.29,50.29,45.39,38.32,32.42,26.18};

    double d_0[8] = {1051.97,1132.19,1258.05,1511.55,1695.9,2059.74,2486.69,3160.74};
    double d_1[8] = {250.368,248.8,238.24,233.856,239.936,256.608,292.576,362.88};
    double d_2[8] = {423.296,462.272,510.208,590.368,672.16,828.48,999.936,1270.91};
    double d_3[8] = {151.552,157.376,164.192,193.504,216.448,254.016,300.672,374.496};
    double d_4[8] = {422.976,461.696,509.024,588.992,671.136,824.992,997.568,1269.98};
    double d_5[8] = {161.216,160.352,178.176,194.848,210.208,258.912,293.984,369.248};
    double d_6[8] = {2534.69,2758.18,3045.5,3536.74,4007.65,4912,5941.28,7623.84};
    double d_7[8] = {498.176,481.44,463.136,441.44,447.36,480.96,558.656,701.28};
    double d_8[8] = {604.32,585.152,579.808,590.752,668.064,822.208,993.376,1261.66};
    double d_9[8] = {59.04,64.352,57.92,197.984,217.024,258.4,306.656,369.632};
    sgem.init(d_0,b_0);
    prediction.emplace(key_0,sgem);
    sgem.init(d_1,b_1);
    prediction.emplace(key_1,sgem);
    sgem.init(d_2,b_2);
    prediction.emplace(key_2,sgem);
    sgem.init(d_3,b_3);
    prediction.emplace(key_3,sgem);
    sgem.init(d_4,b_4);
    prediction.emplace(key_4,sgem);
    sgem.init(d_5,b_5);
    prediction.emplace(key_5,sgem);
    sgem.init(d_6,b_6);
    prediction.emplace(key_6,sgem);
    sgem.init(d_7,b_7);
    prediction.emplace(key_7,sgem);
    sgem.init(d_8,b_8);
    prediction.emplace(key_8,sgem);
    sgem.init(d_9,b_9);
    prediction.emplace(key_9,sgem);
}

// prediction.emplace_back(48214,);
// prediction.emplace_back(47886,);
// prediction.emplace_back(47974,);
// prediction.emplace_back(47121,);
// prediction.emplace_back(47968,);
// prediction.emplace_back(47112,);

CublasInterProcess cublas_inter_process;
static pthread_mutex_t api_used = PTHREAD_MUTEX_INITIALIZER;
std::uintptr_t tmp_ptr;
static std::unordered_map<std::uintptr_t, cudaIpcMemHandle_st> if_ptr_open;
CUBLAS_TYPE cublas_type;
SgemmInfo sgemm_info;
static bool if_first = true;
int tmp_key;
// double bandwidth[11];
// double duration[11];
void get_memhandle(std::uintptr_t &tmp_ptr_,
                   cudaIpcMemHandle_t &tmp_memhandle_,
                   std::unordered_map<std::uintptr_t, cudaIpcMemHandle_t> &if_ptr_open_)
{
    if (if_ptr_open_.find(tmp_ptr_) != if_ptr_open_.end())
    {
        tmp_memhandle_ = if_ptr_open_[tmp_ptr_];
    }
    else
    {
        CUDA_CHECK(cudaIpcGetMemHandle(&tmp_memhandle_, (void *)tmp_ptr_));
        if_ptr_open_.emplace(tmp_ptr_, tmp_memhandle_);
    }
}
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
    pthread_mutex_lock(&api_used);
    if (if_first)
    {
        init_prediction();
        if_first = false;
    }
#ifndef NDEBUG
    std::cout << "cublasSgemm hooked" << std::endl;
#endif
    cublas_type = SGEMM;
    sgemm_info.transa = transa;
    sgemm_info.transb = transb;
    sgemm_info.m = m;
    sgemm_info.n = n;
    sgemm_info.k = k;
    sgemm_info.alpha = *alpha;
    sgemm_info.beta = *beta;
    sgemm_info.A_devptr = (std::uintptr_t)A;
    get_memhandle(sgemm_info.A_devptr, sgemm_info.A_handle, if_ptr_open);
    sgemm_info.B_devptr = (std::uintptr_t)B;
    get_memhandle(sgemm_info.B_devptr, sgemm_info.B_handle, if_ptr_open);
    sgemm_info.C_devptr = (std::uintptr_t)C;
    get_memhandle(sgemm_info.C_devptr, sgemm_info.C_handle, if_ptr_open);
    sgemm_info.lda = lda;
    sgemm_info.ldb = ldb;
    sgemm_info.ldc = ldc;
    tmp_key = m + n + k + lda + ldb + ldc;
    // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    cublas_inter_process.send_cublas(cublas_type, sgemm_info, prediction[tmp_key].duration, prediction[tmp_key].bandwidth);
    if (sem_trywait(&cublas_inter_process.shm_cublas->sync.percent_sem[0]) == 0)
    {
        __typeof__(cublasSgemm_v2) *fp = (__typeof__(cublasSgemm_v2) *)dlsym(RTLD_NEXT, "cublasSgemm_v2");
        cublasStatus_t ret = fp(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        if (sem_trywait(&cublas_inter_process.shm_cublas->sync.if_sync) == 0)
        {
            CUDA_CHECK(cudaDeviceSynchronize());
            sem_post(&cublas_inter_process.shm_cublas->sync.synced);
        }
        pthread_mutex_unlock(&api_used);
        return ret;
    }
    pthread_mutex_unlock(&api_used);
    return CUBLAS_STATUS_SUCCESS;
}