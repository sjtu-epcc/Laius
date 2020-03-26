/*
 * File: /home/haohao/Projects/Paper/reference/cudnn/src/cudnn/cudnn_api.cc
 * Project: /home/haohao/Projects/Paper/reference/cudnn
 * Created Date: Friday, December 21st 2018, 3:36:50 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Monday, January 28th 2019, 3:11:07 pm
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 *
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include "cudnn/cudnn_api.hpp"

void api_server(cudnnHandle_t *cudnn,
                ShmCompute *shm_compute, ShmSet *shm_set,
                DescriptorStore &desc_store, DeviceStore &dev_store,
                int &cur_percent_, int &cur_pid_)
{
    AddTensor addtensor_solver;
    ActivationFwd activation_fwd_solver;
    ConvFwd conv_fwd_solver;
    PoolingFwd pooling_fwd_solver;
    SoftmaxFwd softmax_fwd_solver;
    int compute_cnt = 0;
    int compute_total = 0;
    DLOG(INFO) << "Wait for Cudnn API!";
    while (true)
    {
        if (sem_trywait(&shm_set->sync.sem_ready) == 0)
        {
            switch (shm_set->set_call.api_type)
            {
            case SETTENSOR4D:
            {
                DLOG(INFO) << "Set tensor4d received";
                cudnnTensorDescriptor_t tmp_tensor;
                CUDNN_CHECK(cudnnCreateTensorDescriptor(&tmp_tensor));
                CUDNN_CHECK(cudnnSetTensor4dDescriptor(tmp_tensor,
                                                       shm_set->set_call.set_tensor4d.format,
                                                       shm_set->set_call.set_tensor4d.dataType,
                                                       shm_set->set_call.set_tensor4d.n,
                                                       shm_set->set_call.set_tensor4d.c,
                                                       shm_set->set_call.set_tensor4d.h,
                                                       shm_set->set_call.set_tensor4d.w));
                // CUDA_CHECK(cudaDeviceSynchronize());
                pthread_mutex_lock(&desc_store.mutex);
                if (desc_store.tensor_store.find(shm_set->set_call.set_tensor4d.desc_name) == desc_store.tensor_store.end())
                    CHECK_EQ(desc_store.tensor_store.emplace(shm_set->set_call.set_tensor4d.desc_name, tmp_tensor).second, true);
                else
                    desc_store.tensor_store[shm_set->set_call.set_tensor4d.desc_name] = tmp_tensor;
                pthread_mutex_unlock(&desc_store.mutex);
                break;
            }
            case SETTENSOR4DEX:
            {
                DLOG(INFO) << "Set tensor4dex received";
                cudnnTensorDescriptor_t tmp_tensor;
                CUDNN_CHECK(cudnnCreateTensorDescriptor(&tmp_tensor));
                CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(tmp_tensor,
                                                         shm_set->set_call.set_tensor4dex.dataType,
                                                         shm_set->set_call.set_tensor4dex.n,
                                                         shm_set->set_call.set_tensor4dex.c,
                                                         shm_set->set_call.set_tensor4dex.h,
                                                         shm_set->set_call.set_tensor4dex.w,
                                                         shm_set->set_call.set_tensor4dex.nStride,
                                                         shm_set->set_call.set_tensor4dex.cStride,
                                                         shm_set->set_call.set_tensor4dex.hStride,
                                                         shm_set->set_call.set_tensor4dex.wStride));
                // CUDA_CHECK(cudaDeviceSynchronize());
                pthread_mutex_lock(&desc_store.mutex);
                if (desc_store.tensor_store.find(shm_set->set_call.set_tensor4dex.desc_name) == desc_store.tensor_store.end())
                    CHECK_EQ(desc_store.tensor_store.emplace(shm_set->set_call.set_tensor4dex.desc_name, tmp_tensor).second, true);
                else
                    desc_store.tensor_store[shm_set->set_call.set_tensor4dex.desc_name] = tmp_tensor;
                pthread_mutex_unlock(&desc_store.mutex);
                break;
            }
            case SETCONV2D:
            {
                DLOG(INFO) << "Set conv2d received";
                cudnnConvolutionDescriptor_t tmp_conv;
                CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&tmp_conv));
                CUDNN_CHECK(cudnnSetConvolution2dDescriptor(tmp_conv,
                                                            shm_set->set_call.set_conv2d.pad_h,
                                                            shm_set->set_call.set_conv2d.pad_w,
                                                            shm_set->set_call.set_conv2d.u,
                                                            shm_set->set_call.set_conv2d.v,
                                                            shm_set->set_call.set_conv2d.dilation_h,
                                                            shm_set->set_call.set_conv2d.dilation_w,
                                                            shm_set->set_call.set_conv2d.mode,
                                                            shm_set->set_call.set_conv2d.computeType));
                // CUDA_CHECK(cudaDeviceSynchronize());
                pthread_mutex_lock(&desc_store.mutex);
                if (desc_store.conv_store.find(shm_set->set_call.set_conv2d.desc_name) == desc_store.conv_store.end())
                    CHECK_EQ(desc_store.conv_store.emplace(shm_set->set_call.set_conv2d.desc_name, tmp_conv).second, true);
                else
                    desc_store.conv_store[shm_set->set_call.set_conv2d.desc_name] = tmp_conv;
                pthread_mutex_unlock(&desc_store.mutex);
                break;
            }
            case SETFILTER4D:
            {
                DLOG(INFO) << "Set filter4d received";
                cudnnFilterDescriptor_t tmp_filter;
                CUDNN_CHECK(cudnnCreateFilterDescriptor(&tmp_filter));
                CUDNN_CHECK(cudnnSetFilter4dDescriptor(tmp_filter,
                                                       shm_set->set_call.set_filter4d.dataType,
                                                       shm_set->set_call.set_filter4d.format,
                                                       shm_set->set_call.set_filter4d.k,
                                                       shm_set->set_call.set_filter4d.c,
                                                       shm_set->set_call.set_filter4d.h,
                                                       shm_set->set_call.set_filter4d.w));
                // CUDA_CHECK(cudaDeviceSynchronize());
                pthread_mutex_lock(&desc_store.mutex);
                if (desc_store.filter_store.find(shm_set->set_call.set_filter4d.desc_name) == desc_store.filter_store.end())
                    CHECK_EQ(desc_store.filter_store.emplace(shm_set->set_call.set_filter4d.desc_name, tmp_filter).second, true);
                else
                    desc_store.filter_store[shm_set->set_call.set_filter4d.desc_name] = tmp_filter;
                pthread_mutex_unlock(&desc_store.mutex);
                break;
            }
            case SETACTIVATION:
            {
                DLOG(INFO) << "Set activation received";
                cudnnActivationDescriptor_t tmp_activation;
                CUDNN_CHECK(cudnnCreateActivationDescriptor(&tmp_activation));
                CUDNN_CHECK(cudnnSetActivationDescriptor(tmp_activation,
                                                         shm_set->set_call.set_activation.mode,
                                                         shm_set->set_call.set_activation.reluNanOpt,
                                                         shm_set->set_call.set_activation.coef));
                // CUDA_CHECK(cudaDeviceSynchronize());
                pthread_mutex_lock(&desc_store.mutex);
                if (desc_store.activation_store.find(shm_set->set_call.set_activation.desc_name) == desc_store.activation_store.end())
                    CHECK_EQ(desc_store.activation_store.emplace(shm_set->set_call.set_activation.desc_name, tmp_activation).second, true);
                else
                    desc_store.activation_store[shm_set->set_call.set_activation.desc_name] = tmp_activation;
                pthread_mutex_unlock(&desc_store.mutex);
                break;
            }
            case SETPOOLING2D:
            {
                DLOG(INFO) << "Set pooling2d received";
                cudnnPoolingDescriptor_t tmp_pooling;
                CUDNN_CHECK(cudnnCreatePoolingDescriptor(&tmp_pooling));
                CUDNN_CHECK(cudnnSetPooling2dDescriptor(tmp_pooling,
                                                        shm_set->set_call.set_pooling2d.mode,
                                                        shm_set->set_call.set_pooling2d.maxpoolingNanOpt,
                                                        shm_set->set_call.set_pooling2d.windowHeight,
                                                        shm_set->set_call.set_pooling2d.windowWidth,
                                                        shm_set->set_call.set_pooling2d.verticalPadding,
                                                        shm_set->set_call.set_pooling2d.horizontalPadding,
                                                        shm_set->set_call.set_pooling2d.verticalStride,
                                                        shm_set->set_call.set_pooling2d.horizontalStride));
                // CUDA_CHECK(cudaDeviceSynchronize());
                pthread_mutex_lock(&desc_store.mutex);
                if (desc_store.pooling_store.find(shm_set->set_call.set_pooling2d.desc_name) == desc_store.pooling_store.end())
                    CHECK_EQ(desc_store.pooling_store.emplace(shm_set->set_call.set_pooling2d.desc_name, tmp_pooling).second, true);
                else
                    desc_store.pooling_store[shm_set->set_call.set_pooling2d.desc_name] = tmp_pooling;
                pthread_mutex_unlock(&desc_store.mutex);
                break;
            }
            }
            pthread_barrier_wait(&shm_set->sync.barrier);
        }
        if (sem_trywait(&shm_compute->sync.percent_sem[cur_percent_]) == 0)
        {
            compute_cnt++;
            compute_total++;

            DLOG(INFO) << "------" << compute_total << "-th api computed------"
                       << " Using " << cur_percent_ * 10 << " percentage";
            // shm_compute->if_computed = true;
            // DLOG(INFO) << shm_compute[i]->cudnn_call.api_type<< std::endl;
            switch (shm_compute->cudnn_call.api_type)
            {
            case ADDTENSOR:
                DLOG(INFO) << "ADDTENSOR received";
                addtensor_solver.setup(shm_compute, &desc_store, &dev_store);
                addtensor_solver.compute(cudnn);
                break;
            case CONVFWD:
                DLOG(INFO) << "CONVFWD received";
                conv_fwd_solver.setup(shm_compute, &desc_store, &dev_store);
                conv_fwd_solver.compute(cudnn);
                break;
            case ACTIVATIONFWD:
                DLOG(INFO) << "ACTIVATIONFWD received";
                activation_fwd_solver.setup(shm_compute, &desc_store, &dev_store);
                activation_fwd_solver.compute(cudnn);
                break;
            case POOLFWD:
                DLOG(INFO) << "POOLFWD received";
                pooling_fwd_solver.setup(shm_compute, &desc_store, &dev_store);
                pooling_fwd_solver.compute(cudnn);
                break;
            case SOFTMAXFWD:
                DLOG(INFO) << "SOFTMAXFWD received";
                softmax_fwd_solver.setup(shm_compute, &desc_store, &dev_store);
                softmax_fwd_solver.compute(cudnn);
                break;
            default:
                DLOG(INFO) << "No such cudnn api" << std::endl;
                std::abort();
                break;
            }
            if (sem_trywait(&shm_compute->sync.if_sync) == 0)
            {
                CUDA_CHECK(cudaDeviceSynchronize());
                sem_post(&shm_compute->sync.synced);
                DLOG(INFO) << "------synced------";
            }
            pthread_barrier_wait(&shm_compute->sync.barrier);
        }
    }
}

void *compute_server(void *compute_args_)
{
    ComputeArgs *compute_args = (ComputeArgs *)compute_args_;
    AddTensor addtensor_solver;
    ActivationFwd activation_fwd_solver;
    ConvFwd conv_fwd_solver;
    PoolingFwd pooling_fwd_solver;
    SoftmaxFwd softmax_fwd_solver;
    int api_cnt = 0;
    int api_total = 0;
    while (true)
    {
        // DLOG(INFO) << "------SelfPercent: " << *compute_args->cur_percent_ * 10;
        //    << "------SharedPercent: " << compute_args->shm_compute->percent_flag
        //    << "------Process: " << compute_args->shm_compute->process_flag << "------";
        if (sem_trywait(&compute_args->shm_compute->sync.percent_sem[*compute_args->cur_percent_]) == 0)
        {
            api_cnt++;
            api_total++;

//            DLOG(INFO) << "------" << api_total << "-th api computed------"
//                       << " Using " << *compute_args->cur_percent_ * 10 << " percentage";
            // compute_args->shm_compute->if_computed = true;
            // DLOG(INFO) << shm_compute[i]->cudnn_call.api_type<< std::endl;
            switch (compute_args->shm_compute->cudnn_call.api_type)
            {
            case ADDTENSOR:
//                DLOG(INFO) << "ADDTENSOR received";
                addtensor_solver.setup(compute_args->shm_compute, compute_args->descriptor_store, compute_args->device_store);
                addtensor_solver.compute(compute_args->cudnn);
                break;
            case CONVFWD:
//                DLOG(INFO) << "CONVFWD received";
                conv_fwd_solver.setup(compute_args->shm_compute, compute_args->descriptor_store, compute_args->device_store);
                conv_fwd_solver.compute(compute_args->cudnn);
                break;
            case ACTIVATIONFWD:
//                DLOG(INFO) << "ACTIVATIONFWD received";
                activation_fwd_solver.setup(compute_args->shm_compute, compute_args->descriptor_store, compute_args->device_store);
                activation_fwd_solver.compute(compute_args->cudnn);
                break;
            case POOLFWD:
//                DLOG(INFO) << "POOLFWD received";
                pooling_fwd_solver.setup(compute_args->shm_compute, compute_args->descriptor_store, compute_args->device_store);
                pooling_fwd_solver.compute(compute_args->cudnn);
                break;
            case SOFTMAXFWD:
//                DLOG(INFO) << "SOFTMAXFWD received";
                softmax_fwd_solver.setup(compute_args->shm_compute, compute_args->descriptor_store, compute_args->device_store);
                softmax_fwd_solver.compute(compute_args->cudnn);
                break;
            default:
                DLOG(INFO) << "No such cudnn api" << std::endl;
            }
            if (sem_trywait(&compute_args->shm_compute->sync.if_sync) == 0)
            {
                CUDA_CHECK(cudaDeviceSynchronize());
                sem_post(&compute_args->shm_compute->sync.synced);
//                DLOG(INFO) << "------synced------";
            }
            //     while (compute_args->shm_compute->process_flag != 3)
            pthread_barrier_wait(&compute_args->shm_compute->sync.barrier);
        }
        else
            continue;
    }
}

void *set_server(void *set_args_)
{
    SetArgs *set_args = (SetArgs *)set_args_;
    while (true)
    {
        // DLOG(INFO) << "Process " << getpid() << " is at line " << __LINE__;
        pthread_barrier_wait(&set_args->shm_set->sync.barrier);
        switch (set_args->shm_set->set_call.api_type)
        {
        case SETTENSOR4D:
        {
            DLOG(INFO) << "Set tensor4d received";
            cudnnTensorDescriptor_t tmp_tensor;
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&tmp_tensor));
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(tmp_tensor,
                                                   set_args->shm_set->set_call.set_tensor4d.format,
                                                   set_args->shm_set->set_call.set_tensor4d.dataType,
                                                   set_args->shm_set->set_call.set_tensor4d.n,
                                                   set_args->shm_set->set_call.set_tensor4d.c,
                                                   set_args->shm_set->set_call.set_tensor4d.h,
                                                   set_args->shm_set->set_call.set_tensor4d.w));
            CUDA_CHECK(cudaDeviceSynchronize());
            pthread_mutex_lock(&set_args->descriptor_store->mutex);
            if (set_args->descriptor_store->tensor_store.find(set_args->shm_set->set_call.set_tensor4d.desc_name) == set_args->descriptor_store->tensor_store.end())
                CHECK_EQ(set_args->descriptor_store->tensor_store
                             .emplace(set_args->shm_set->set_call
                                          .set_tensor4d.desc_name,
                                      tmp_tensor)
                             .second,
                         true);
            else
                set_args->descriptor_store->tensor_store[set_args->shm_set->set_call.set_tensor4d.desc_name] = tmp_tensor;
            pthread_mutex_unlock(&set_args->descriptor_store->mutex);
            pthread_barrier_wait(&set_args->shm_set->sync.barrier);
            break;
        }
        case SETTENSOR4DEX:
        {
            DLOG(INFO) << "Set tensor4dex received";
            cudnnTensorDescriptor_t tmp_tensor;
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&tmp_tensor));
            CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(tmp_tensor,
                                                     set_args->shm_set->set_call.set_tensor4dex.dataType,
                                                     set_args->shm_set->set_call.set_tensor4dex.n,
                                                     set_args->shm_set->set_call.set_tensor4dex.c,
                                                     set_args->shm_set->set_call.set_tensor4dex.h,
                                                     set_args->shm_set->set_call.set_tensor4dex.w,
                                                     set_args->shm_set->set_call.set_tensor4dex.nStride,
                                                     set_args->shm_set->set_call.set_tensor4dex.cStride,
                                                     set_args->shm_set->set_call.set_tensor4dex.hStride,
                                                     set_args->shm_set->set_call.set_tensor4dex.wStride));
            CUDA_CHECK(cudaDeviceSynchronize());
            pthread_mutex_lock(&set_args->descriptor_store->mutex);
            if (set_args->descriptor_store->tensor_store.find(set_args->shm_set->set_call.set_tensor4dex.desc_name) == set_args->descriptor_store->tensor_store.end())
                CHECK_EQ(set_args->descriptor_store->tensor_store
                             .emplace(set_args->shm_set->set_call
                                          .set_tensor4dex.desc_name,
                                      tmp_tensor)
                             .second,
                         true);
            else
                set_args->descriptor_store->tensor_store[set_args->shm_set->set_call.set_tensor4dex.desc_name] = tmp_tensor;
            pthread_mutex_unlock(&set_args->descriptor_store->mutex);
            pthread_barrier_wait(&set_args->shm_set->sync.barrier);
            break;
        }
        case SETCONV2D:
        {
            DLOG(INFO) << "Set conv2d received";
            cudnnConvolutionDescriptor_t tmp_conv;
            CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&tmp_conv));
            CUDNN_CHECK(cudnnSetConvolution2dDescriptor(tmp_conv,
                                                        set_args->shm_set->set_call.set_conv2d.pad_h,
                                                        set_args->shm_set->set_call.set_conv2d.pad_w,
                                                        set_args->shm_set->set_call.set_conv2d.u,
                                                        set_args->shm_set->set_call.set_conv2d.v,
                                                        set_args->shm_set->set_call.set_conv2d.dilation_h,
                                                        set_args->shm_set->set_call.set_conv2d.dilation_w,
                                                        set_args->shm_set->set_call.set_conv2d.mode,
                                                        set_args->shm_set->set_call.set_conv2d.computeType));
            CUDA_CHECK(cudaDeviceSynchronize());
            pthread_mutex_lock(&set_args->descriptor_store->mutex);
            if (set_args->descriptor_store->conv_store.find(set_args->shm_set->set_call.set_conv2d.desc_name) == set_args->descriptor_store->conv_store.end())
                CHECK_EQ(set_args->descriptor_store->conv_store
                             .emplace(set_args->shm_set->set_call
                                          .set_conv2d.desc_name,
                                      tmp_conv)
                             .second,
                         true);
            else
                set_args->descriptor_store->conv_store[set_args->shm_set->set_call.set_conv2d.desc_name] = tmp_conv;
            pthread_mutex_unlock(&set_args->descriptor_store->mutex);
            pthread_barrier_wait(&set_args->shm_set->sync.barrier);
            break;
        }
        case SETFILTER4D:
        {
            DLOG(INFO) << "Set filter4d received";
            cudnnFilterDescriptor_t tmp_filter;
            CUDNN_CHECK(cudnnCreateFilterDescriptor(&tmp_filter));
            CUDNN_CHECK(cudnnSetFilter4dDescriptor(tmp_filter,
                                                   set_args->shm_set->set_call.set_filter4d.dataType,
                                                   set_args->shm_set->set_call.set_filter4d.format,
                                                   set_args->shm_set->set_call.set_filter4d.k,
                                                   set_args->shm_set->set_call.set_filter4d.c,
                                                   set_args->shm_set->set_call.set_filter4d.h,
                                                   set_args->shm_set->set_call.set_filter4d.w));
            CUDA_CHECK(cudaDeviceSynchronize());
            pthread_mutex_lock(&set_args->descriptor_store->mutex);
            if (set_args->descriptor_store->filter_store.find(set_args->shm_set->set_call.set_filter4d.desc_name) == set_args->descriptor_store->filter_store.end())
                CHECK_EQ(set_args->descriptor_store->filter_store
                             .emplace(set_args->shm_set->set_call
                                          .set_filter4d.desc_name,
                                      tmp_filter)
                             .second,
                         true);
            else
                set_args->descriptor_store->filter_store[set_args->shm_set->set_call.set_filter4d.desc_name] = tmp_filter;
            pthread_mutex_unlock(&set_args->descriptor_store->mutex);
            pthread_barrier_wait(&set_args->shm_set->sync.barrier);
            break;
        }
        case SETACTIVATION:
        {
            DLOG(INFO) << "Set activation received";
            cudnnActivationDescriptor_t tmp_activation;
            CUDNN_CHECK(cudnnCreateActivationDescriptor(&tmp_activation));
            CUDNN_CHECK(cudnnSetActivationDescriptor(tmp_activation,
                                                     set_args->shm_set->set_call.set_activation.mode,
                                                     set_args->shm_set->set_call.set_activation.reluNanOpt,
                                                     set_args->shm_set->set_call.set_activation.coef));
            CUDA_CHECK(cudaDeviceSynchronize());
            pthread_mutex_lock(&set_args->descriptor_store->mutex);
            if (set_args->descriptor_store->activation_store.find(set_args->shm_set->set_call.set_activation.desc_name) == set_args->descriptor_store->activation_store.end())
                CHECK_EQ(set_args->descriptor_store->activation_store
                             .emplace(set_args->shm_set->set_call
                                          .set_activation.desc_name,
                                      tmp_activation)
                             .second,
                         true);
            else
                set_args->descriptor_store->activation_store[set_args->shm_set->set_call.set_activation.desc_name] = tmp_activation;
            pthread_mutex_unlock(&set_args->descriptor_store->mutex);
            pthread_barrier_wait(&set_args->shm_set->sync.barrier);
            break;
        }
        case SETPOOLING2D:
        {
            DLOG(INFO) << "Set pooling2d received";
            cudnnPoolingDescriptor_t tmp_pooling;
            CUDNN_CHECK(cudnnCreatePoolingDescriptor(&tmp_pooling));
            CUDNN_CHECK(cudnnSetPooling2dDescriptor(tmp_pooling,
                                                    set_args->shm_set->set_call.set_pooling2d.mode,
                                                    set_args->shm_set->set_call.set_pooling2d.maxpoolingNanOpt,
                                                    set_args->shm_set->set_call.set_pooling2d.windowHeight,
                                                    set_args->shm_set->set_call.set_pooling2d.windowWidth,
                                                    set_args->shm_set->set_call.set_pooling2d.verticalPadding,
                                                    set_args->shm_set->set_call.set_pooling2d.horizontalPadding,
                                                    set_args->shm_set->set_call.set_pooling2d.verticalStride,
                                                    set_args->shm_set->set_call.set_pooling2d.horizontalStride));
            CUDA_CHECK(cudaDeviceSynchronize());
            pthread_mutex_lock(&set_args->descriptor_store->mutex);
            if (set_args->descriptor_store->pooling_store.find(set_args->shm_set->set_call.set_pooling2d.desc_name) == set_args->descriptor_store->pooling_store.end())
                CHECK_EQ(set_args->descriptor_store->pooling_store
                             .emplace(set_args->shm_set->set_call
                                          .set_pooling2d.desc_name,
                                      tmp_pooling)
                             .second,
                         true);
            else
                set_args->descriptor_store->pooling_store[set_args->shm_set->set_call.set_pooling2d.desc_name] = tmp_pooling;
            pthread_mutex_unlock(&set_args->descriptor_store->mutex);
            pthread_barrier_wait(&set_args->shm_set->sync.barrier);
            break;
        }
        }
    }
}
