/*
 * File: /home/haohao/Projects/Paper/reference/mps/src/optimized/wrap.cc
 * Project: /home/haohao/Projects/Paper/reference/mps
 * Created Date: Tuesday, December 18th 2018, 7:32:22 pm
 * Author: Raphael-Hao
 * -----
 * Last Modified: Tuesday, January 8th 2019, 10:58:59 am
 * Modified By: Raphael-Hao
 * -----
 * Copyright (c) 2018 Happy
 * 
 * Were It to Benefit My Country, I Would Lay Down My Life !
 */
#include <dlfcn.h>
#include <iostream>
#include <cstdint>
#include <cudnn.h>
#include <pthread.h>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include "optimized/cudnn_ipc.hpp"
#include "check.hpp"

struct predict
{
    double duration[11];
    double bandwidth[11];
    void init()
    {
        memset(duration, 0, 11 * sizeof(double));
        memset(bandwidth, 0, 11 * sizeof(double));
    }
};
void CsvToMap(std::ifstream &time_input_, std::ifstream &band_input_, std::unordered_map<std::vector<int>, predict, container_hash<std::vector<int>>> &dict)
{
    std::string line_input;
    int tmp_precentage;
    DLOG(INFO) << "------Reading Model------";
    while (getline(band_input_, line_input))
    {
        predict tmp_predict;
        tmp_predict.init();
        std::vector<int> int_key;
        std::vector<std::string> str_key;
        std::stringstream key_stream(line_input);
        std::string key_s;
        while (getline(key_stream, key_s, ','))
        {
            str_key.push_back(key_s);
        }
        for (size_t i = 0; i != str_key.size(); i++)
        {
            if (i == (str_key.size() - 2))
            {
                tmp_precentage = atoi(str_key[i].c_str()) / 10;
            }
            else if (i == (str_key.size() - 1))
            {
                tmp_predict.bandwidth[tmp_precentage] = atof(str_key[i].c_str());
                break;
            }
            else
                int_key.push_back(atoi(str_key[i].c_str()));
        }
        if (!dict.emplace(int_key, tmp_predict).second)
            dict[int_key].bandwidth[tmp_precentage] = tmp_predict.bandwidth[tmp_precentage];
    }
    while (getline(time_input_, line_input))
    {
        double elapse;
        std::vector<int> int_key;
        std::vector<std::string> str_key;
        std::stringstream key_stream(line_input);
        std::string key_s;
        while (getline(key_stream, key_s, ','))
        {
            str_key.push_back(key_s);
        }
        for (size_t i = 0; i != str_key.size(); i++)
        {
            if (i == (str_key.size() - 2))
            {
                tmp_precentage = atoi(str_key[i].c_str()) / 10;
            }
            else if (i == (str_key.size() - 1))
            {
                elapse = atof(str_key[i].c_str());
            }
            else
                int_key.push_back(atoi(str_key[i].c_str()));
        }
        if (dict.count(int_key) == 1)
        {
            dict[int_key].duration[tmp_precentage] = elapse;
        }
        else
        {
            std::cout << "time data and bandwidth data no match" << std::endl;
            std::abort();
        }
    }
}

static SetInterProcess set_inter_process;
static ComputeInterProcess compute_inter_process;
static pthread_mutex_t api_used = PTHREAD_MUTEX_INITIALIZER;
static std::unordered_map<uintptr_t, int> if_ptr_open;
static std::unordered_map<std::vector<int>, predict, container_hash<std::vector<int>>> addtensor_dict;
static std::unordered_map<std::vector<int>, predict, container_hash<std::vector<int>>> convolution_forward_dict;
static std::unordered_map<std::vector<int>, predict, container_hash<std::vector<int>>> activation_forward_dict;
static std::unordered_map<std::vector<int>, predict, container_hash<std::vector<int>>> pooling_forward_dict;
static std::unordered_map<std::vector<int>, predict, container_hash<std::vector<int>>> softmax_forward_dict;
static bool first_hook = true;
// static int percentage_flag = 0;

void load_model()
{
    std::ifstream time_input;
    std::ifstream band_input;
    std::string data_path = getenv("PREDICT_DATA_PATH");
    // percentage_flag = atoi(getenv("CURRENT_PERCENTAGE"));
    time_input.open(data_path + "/cudnnAddTensor_t.csv", std::ios::in);
    band_input.open(data_path + "/cudnnAddTensor_b.csv", std::ios::in);
    CsvToMap(time_input, band_input, addtensor_dict);
    time_input.close();
    time_input.clear();
    band_input.close();
    band_input.clear();

    time_input.open(data_path + "/cudnnConvolutionForward_t.csv", std::ios::in);
    band_input.open(data_path + "/cudnnConvolutionForward_b.csv", std::ios::in);
    CsvToMap(time_input, band_input, convolution_forward_dict);
    time_input.close();
    time_input.clear();
    band_input.close();
    band_input.clear();

    time_input.open(data_path + "/cudnnActivationForward_t.csv", std::ios::in);
    band_input.open(data_path + "/cudnnActivationForward_b.csv", std::ios::in);
    CsvToMap(time_input, band_input, activation_forward_dict);
    time_input.close();
    time_input.clear();
    band_input.close();
    band_input.clear();

    time_input.open(data_path + "/cudnnPoolingForward_b.csv", std::ios::in);
    band_input.open(data_path + "/cudnnPoolingForward_t.csv", std::ios::in);
    CsvToMap(time_input, band_input, pooling_forward_dict);
    time_input.close();
    time_input.clear();
    band_input.close();
    band_input.clear();

    time_input.open(data_path + "/cudnnSoftmaxForward_b.csv", std::ios::in);
    band_input.open(data_path + "/cudnnSoftmaxForward_t.csv", std::ios::in);
    CsvToMap(time_input, band_input, softmax_forward_dict);
    time_input.close();
    time_input.clear();
    band_input.close();
    band_input.clear();
}

uintptr_t tmp_if_ptr;
COMPUTE_TYPE compute_type;
SET_TYPE set_type;
TensorInfo tensor_in_info;
TensorInfo tensor_out_info;
ActivationFwdInfo activation_fwd_info;
ConvFwdInfo conv_fwd_info;
PoolingFwdInfo pooling_fwd_info;
SoftmaxFwdInfo softmax_fwd_info;
SetTensor4d set_tensor4d;
SetTensor4dex set_tensor4dex;
SetConv2d set_conv2d;
SetFilter4d set_filter4d;
SetActivation set_activation;
SetPooling2d set_pooling2d;
float Alpha, Beta;

struct TensorShape
{
    cudnnDataType_t type;
    int n, c, h, w, nStride, cStride, hStride, wStride;
} tensor_shape;
struct Conv2dShape
{
    cudnnConvolutionMode_t mode;
    cudnnDataType_t type;
    int pad_h, pad_w, u, v, dilation_h, dilation_w;
} conv2d_shape;
struct FilterShape
{
    cudnnDataType_t type;
    cudnnTensorFormat_t format;
    int k, c, h, w;
} filter_shape;
struct ActivationShape
{
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t reluNanOpt;
    double coef;
} activation_shape;
struct PoolingShape
{
    cudnnPoolingMode_t mode;
    cudnnNanPropagation_t maxpoolingNanOpt;
    int wHeight, wWidth, vPadding, hPadding, vStride, hStride;
} pooling_shape;

static std::unordered_map<uintptr_t, TensorShape> tensor_all;
static std::unordered_map<uintptr_t, Conv2dShape> conv2d_all;
static std::unordered_map<uintptr_t, FilterShape> filter_all;
static std::unordered_map<uintptr_t, ActivationShape> activation_all;
static std::unordered_map<uintptr_t, PoolingShape> pooling_all;

void tensor_emplace(std::vector<int> &add_key_, uintptr_t &tensor_name_)
{
    add_key_.emplace_back(tensor_all[tensor_name_].type);
    add_key_.emplace_back(tensor_all[tensor_name_].n);
    add_key_.emplace_back(tensor_all[tensor_name_].c);
    add_key_.emplace_back(tensor_all[tensor_name_].h);
    add_key_.emplace_back(tensor_all[tensor_name_].w);
    add_key_.emplace_back(tensor_all[tensor_name_].nStride);
    add_key_.emplace_back(tensor_all[tensor_name_].cStride);
    add_key_.emplace_back(tensor_all[tensor_name_].hStride);
    add_key_.emplace_back(tensor_all[tensor_name_].wStride);
}

void convolution_fwd_emplace(std::vector<int> &add_key_, uintptr_t &filter_name_, uintptr_t conv_name)
{
    add_key_.emplace_back(filter_all[filter_name_].type);
    add_key_.emplace_back(filter_all[filter_name_].format);
    add_key_.emplace_back(filter_all[filter_name_].k);
    add_key_.emplace_back(filter_all[filter_name_].c);
    add_key_.emplace_back(filter_all[filter_name_].h);
    add_key_.emplace_back(filter_all[filter_name_].w);
    add_key_.emplace_back(conv2d_all[conv_name].pad_h);
    add_key_.emplace_back(conv2d_all[conv_name].pad_w);
    add_key_.emplace_back(conv2d_all[conv_name].u);
    add_key_.emplace_back(conv2d_all[conv_name].v);
    add_key_.emplace_back(conv2d_all[conv_name].dilation_h);
    add_key_.emplace_back(conv2d_all[conv_name].dilation_w);
    add_key_.emplace_back(conv2d_all[conv_name].mode);
    add_key_.emplace_back(conv2d_all[conv_name].type);
}

void activation_fwd_emplace(std::vector<int> &add_key_, uintptr_t activation_name_)
{
    add_key_.emplace_back(activation_all[activation_name_].mode);
    add_key_.emplace_back(activation_all[activation_name_].reluNanOpt);
    add_key_.emplace_back(activation_all[activation_name_].coef);
}

void pooling_fwd_emplace(std::vector<int> &add_key_, uintptr_t pooling_name_)
{
    add_key_.emplace_back(pooling_all[pooling_name_].mode);
    add_key_.emplace_back(pooling_all[pooling_name_].maxpoolingNanOpt);
    add_key_.emplace_back(pooling_all[pooling_name_].wHeight);
    add_key_.emplace_back(pooling_all[pooling_name_].wWidth);
    add_key_.emplace_back(pooling_all[pooling_name_].vPadding);
    add_key_.emplace_back(pooling_all[pooling_name_].hPadding);
    add_key_.emplace_back(pooling_all[pooling_name_].vStride);
    add_key_.emplace_back(pooling_all[pooling_name_].hStride);
}

extern "C"
{
    cudnnStatus_t CUDNNWINAPI
    cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                               cudnnTensorFormat_t format,
                               cudnnDataType_t dataType, /* image data type */
                               int n,                    /* number of inputs (batch size) */
                               int c,                    /* number of input feature maps */
                               int h,                    /* height of input section */
                               int w)                    /* width of input section */
    {
        std::cout << "Set tensor4d hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        if (first_hook)
        {
            first_hook = false;
            load_model();
        }
        set_type = SETTENSOR4D;
        set_tensor4d.desc_name = (uintptr_t)tensorDesc;
        set_tensor4d.format = format;
        set_tensor4d.dataType = dataType;
        set_tensor4d.n = n;
        set_tensor4d.c = c;
        set_tensor4d.h = h;
        set_tensor4d.w = w;
        set_inter_process.send_set(set_type, set_tensor4d, set_tensor4dex, set_conv2d, set_filter4d, set_activation, set_pooling2d);
        __typeof__(cudnnSetTensor4dDescriptor) *fp = (__typeof__(cudnnSetTensor4dDescriptor) *)dlsym(RTLD_NEXT, "cudnnSetTensor4dDescriptor");
        cudnnStatus_t ret = fp(tensorDesc, format, dataType,
                               n, c, h, w);
        CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc,
                                               &tensor_shape.type,
                                               &tensor_shape.n,
                                               &tensor_shape.c,
                                               &tensor_shape.h,
                                               &tensor_shape.w,
                                               &tensor_shape.nStride,
                                               &tensor_shape.cStride,
                                               &tensor_shape.hStride,
                                               &tensor_shape.wStride));
        if (!tensor_all.emplace((uintptr_t)tensorDesc, tensor_shape).second)
            tensor_all[(uintptr_t)tensorDesc] = tensor_shape;
        pthread_mutex_unlock(&api_used);
        return ret;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                 cudnnDataType_t dataType, /* image data type */
                                 int n,                    /* number of inputs (batch size) */
                                 int c,                    /* number of input feature maps */
                                 int h,                    /* height of input section */
                                 int w,                    /* width of input section */
                                 int nStride,
                                 int cStride,
                                 int hStride,
                                 int wStride)
    {
        std::cout << "Set tensor4dex hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        if (first_hook)
        {
            first_hook = false;
            load_model();
        }
        set_type = SETTENSOR4DEX;
        set_tensor4dex.desc_name = (uintptr_t)tensorDesc;
        set_tensor4dex.dataType = dataType;
        set_tensor4dex.n = n;
        set_tensor4dex.c = c;
        set_tensor4dex.h = h;
        set_tensor4dex.w = w;
        set_tensor4dex.nStride = nStride;
        set_tensor4dex.cStride = cStride;
        set_tensor4dex.hStride = hStride;
        set_tensor4dex.wStride = wStride;
        set_inter_process.send_set(set_type, set_tensor4d, set_tensor4dex, set_conv2d, set_filter4d, set_activation, set_pooling2d);
        __typeof__(cudnnSetTensor4dDescriptorEx) *fp = (__typeof__(cudnnSetTensor4dDescriptorEx) *)dlsym(RTLD_NEXT, "cudnnSetTensor4dDescriptorEx");
        cudnnStatus_t ret = fp(tensorDesc, dataType,
                               n, c, h, w,
                               nStride, cStride, hStride, wStride);
        CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc,
                                               &tensor_shape.type,
                                               &tensor_shape.n,
                                               &tensor_shape.c,
                                               &tensor_shape.h,
                                               &tensor_shape.w,
                                               &tensor_shape.nStride,
                                               &tensor_shape.cStride,
                                               &tensor_shape.hStride,
                                               &tensor_shape.wStride));
        if (!tensor_all.emplace((uintptr_t)tensorDesc, tensor_shape).second)
            tensor_all[(uintptr_t)tensorDesc] = tensor_shape;
        pthread_mutex_unlock(&api_used);
        return ret;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                    int pad_h,      /* zero-padding height */
                                    int pad_w,      /* zero-padding width */
                                    int u,          /* vertical filter stride */
                                    int v,          /* horizontal filter stride */
                                    int dilation_h, /* filter dilation in the vertical dimension */
                                    int dilation_w, /* filter dilation in the horizontal dimension */
                                    cudnnConvolutionMode_t mode,
                                    cudnnDataType_t computeType)
    {
        std::cout << "Set conv2d hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        set_type = SETCONV2D;
        set_conv2d.desc_name = (uintptr_t)convDesc;
        set_conv2d.pad_h = pad_h;
        set_conv2d.pad_w = pad_w;
        set_conv2d.u = u;
        set_conv2d.v = v;
        set_conv2d.dilation_h = dilation_h;
        set_conv2d.dilation_w = dilation_w;
        set_conv2d.mode = mode;
        set_conv2d.computeType = computeType;
        set_inter_process.send_set(set_type, set_tensor4d, set_tensor4dex, set_conv2d, set_filter4d, set_activation, set_pooling2d);
        __typeof__(cudnnSetConvolution2dDescriptor) *fp = (__typeof__(cudnnSetConvolution2dDescriptor) *)dlsym(RTLD_NEXT, "cudnnSetConvolution2dDescriptor");
        cudnnStatus_t ret = fp(convDesc, pad_h, pad_w, u, v,
                               dilation_h, dilation_w, mode, computeType);
        conv2d_shape.pad_h = pad_h;
        conv2d_shape.pad_w = pad_w;
        conv2d_shape.u = u;
        conv2d_shape.v = v;
        conv2d_shape.dilation_h = dilation_h;
        conv2d_shape.dilation_w = dilation_w;
        conv2d_shape.mode = mode;
        conv2d_shape.type = computeType;
        if (!conv2d_all.emplace((uintptr_t)convDesc, conv2d_shape).second)
            conv2d_all[(uintptr_t)convDesc] = conv2d_shape;
        pthread_mutex_unlock(&api_used);
        return ret;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                               cudnnDataType_t dataType, /* image data type */
                               cudnnTensorFormat_t format,
                               int k, /* number of output feature maps */
                               int c, /* number of input feature maps */
                               int h, /* height of each input filter */
                               int w) /* width of  each input filter */
    {
        std::cout << "Set filter4d hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        set_type = SETFILTER4D;
        set_filter4d.desc_name = (uintptr_t)filterDesc;
        set_filter4d.dataType = dataType;
        set_filter4d.format = format;
        set_filter4d.k = k;
        set_filter4d.c = c;
        set_filter4d.h = h;
        set_filter4d.w = w;
        set_inter_process.send_set(set_type, set_tensor4d, set_tensor4dex, set_conv2d, set_filter4d, set_activation, set_pooling2d);
        __typeof__(cudnnSetFilter4dDescriptor) *fp = (__typeof__(cudnnSetFilter4dDescriptor) *)dlsym(RTLD_NEXT, "cudnnSetFilter4dDescriptor");
        cudnnStatus_t ret = fp(filterDesc, dataType,
                               format, k, c, h, w);
        filter_shape.type = dataType;
        filter_shape.format = format;
        filter_shape.k = k;
        filter_shape.c = c;
        filter_shape.h = h;
        filter_shape.w = w;
        if (!filter_all.emplace((uintptr_t)filterDesc, filter_shape).second)
            filter_all[(uintptr_t)filterDesc] = filter_shape;
        pthread_mutex_unlock(&api_used);
        return ret;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                 cudnnActivationMode_t mode,
                                 cudnnNanPropagation_t reluNanOpt,
                                 double coef) /* ceiling for clipped RELU, alpha for ELU */
    {
        std::cout << "Set activation hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        set_type = SETACTIVATION;
        set_activation.desc_name = (uintptr_t)activationDesc;
        set_activation.mode = mode;
        set_activation.reluNanOpt = reluNanOpt;
        set_activation.coef = coef;
        set_inter_process.send_set(set_type, set_tensor4d, set_tensor4dex, set_conv2d, set_filter4d, set_activation, set_pooling2d);
        __typeof__(cudnnSetActivationDescriptor) *fp = (__typeof__(cudnnSetActivationDescriptor) *)dlsym(RTLD_NEXT, "cudnnSetActivationDescriptor");
        cudnnStatus_t ret = fp(activationDesc, mode, reluNanOpt, coef);
        activation_shape.mode = mode;
        activation_shape.reluNanOpt = reluNanOpt;
        activation_shape.coef = coef;
        if (!activation_all.emplace((uintptr_t)activationDesc, activation_shape).second)
            activation_all[(uintptr_t)activationDesc] = activation_shape;
        pthread_mutex_unlock(&api_used);
        return ret;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                cudnnPoolingMode_t mode,
                                cudnnNanPropagation_t maxpoolingNanOpt,
                                int windowHeight,
                                int windowWidth,
                                int verticalPadding,
                                int horizontalPadding,
                                int verticalStride,
                                int horizontalStride)
    {
        std::cout << "Set pooling2d hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        set_type = SETPOOLING2D;
        set_pooling2d.desc_name = (uintptr_t)poolingDesc;
        set_pooling2d.mode = mode;
        set_pooling2d.maxpoolingNanOpt = maxpoolingNanOpt;
        set_pooling2d.windowHeight = windowHeight;
        set_pooling2d.windowWidth = windowWidth;
        set_pooling2d.verticalPadding = verticalPadding;
        set_pooling2d.horizontalPadding = horizontalPadding;
        set_pooling2d.verticalStride = verticalStride;
        set_pooling2d.horizontalStride = horizontalStride;
        set_inter_process.send_set(set_type, set_tensor4d, set_tensor4dex, set_conv2d, set_filter4d, set_activation, set_pooling2d);
        __typeof__(cudnnSetPooling2dDescriptor) *fp = (__typeof__(cudnnSetPooling2dDescriptor) *)dlsym(RTLD_NEXT, "cudnnSetPooling2dDescriptor");
        cudnnStatus_t ret = fp(poolingDesc,
                               mode, maxpoolingNanOpt,
                               windowHeight, windowWidth,
                               verticalPadding, horizontalPadding,
                               verticalStride, horizontalStride);
        pooling_shape.mode = mode;
        pooling_shape.maxpoolingNanOpt = maxpoolingNanOpt;
        pooling_shape.wHeight = windowHeight;
        pooling_shape.wWidth = windowWidth;
        pooling_shape.vPadding = verticalPadding;
        pooling_shape.hPadding = horizontalPadding;
        pooling_shape.vStride = verticalStride;
        pooling_shape.hStride = horizontalStride;
        if (!pooling_all.emplace((uintptr_t)poolingDesc, pooling_shape).second)
            pooling_all[(uintptr_t)poolingDesc] = pooling_shape;
        pthread_mutex_unlock(&api_used);
        return ret;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnAddTensor(cudnnHandle_t handle,
                   const void *alpha,
                   const cudnnTensorDescriptor_t aDesc,
                   const void *A,
                   const void *beta,
                   const cudnnTensorDescriptor_t cDesc,
                   void *C)
    {
        std::cout << "Add tensor hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        compute_type = ADDTENSOR;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        tensor_in_info.tensor_name = (uintptr_t)aDesc;
        tensor_out_info.tensor_name = (uintptr_t)cDesc;
        tmp_if_ptr = (uintptr_t)A;
        tensor_in_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_in_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_in_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_in_info.tensor_handle, (void *)A));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        tmp_if_ptr = (uintptr_t)C;
        tensor_out_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_out_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_out_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_out_info.tensor_handle, C));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        std::vector<int> addtensor_key;
        addtensor_key.emplace_back(Alpha);
        addtensor_key.emplace_back(Beta);
        tensor_emplace(addtensor_key, tensor_in_info.tensor_name);
        tensor_emplace(addtensor_key, tensor_out_info.tensor_name);
        compute_inter_process.send_compute(compute_type, Alpha, Beta,
                                           tensor_in_info, tensor_out_info,
                                           conv_fwd_info, activation_fwd_info,
                                           pooling_fwd_info, softmax_fwd_info,
                                           addtensor_dict[addtensor_key].bandwidth,
                                           addtensor_dict[addtensor_key].duration);
        if (compute_inter_process.compute_need())
        {
            __typeof__(cudnnAddTensor) *addtensor_fp = (__typeof__(cudnnAddTensor) *)dlsym(RTLD_NEXT, "cudnnAddTensor");
            cudnnStatus_t ret = addtensor_fp(handle, alpha, aDesc, A, beta, cDesc, C);
            pthread_mutex_unlock(&api_used);
            return ret;
        }
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnConvolutionForward(cudnnHandle_t handle,
                            const void *alpha,
                            const cudnnTensorDescriptor_t xDesc,
                            const void *x,
                            const cudnnFilterDescriptor_t wDesc,
                            const void *w,
                            const cudnnConvolutionDescriptor_t convDesc,
                            cudnnConvolutionFwdAlgo_t algo,
                            void *workSpace,
                            size_t workSpaceSizeInBytes,
                            const void *beta,
                            const cudnnTensorDescriptor_t yDesc,
                            void *y)
    {
        std::cout << "Convolution fwd hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        compute_type = CONVFWD;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        tensor_in_info.tensor_name = (uintptr_t)xDesc;
        tmp_if_ptr = (uintptr_t)x;
        tensor_in_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_in_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_in_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_in_info.tensor_handle, (void *)x));
            // std::cout << tensor_in_info.tensor_handle.reserved << " " << (uintptr_t)x << std::endl;
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        conv_fwd_info.filter_name = (uintptr_t)wDesc;
        tmp_if_ptr = (uintptr_t)w;
        conv_fwd_info.filter_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            conv_fwd_info.if_filter_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            conv_fwd_info.if_filter_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&conv_fwd_info.filter_handle, (void *)w));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        conv_fwd_info.conv_name = (uintptr_t)convDesc;
        conv_fwd_info.algo = algo;
        conv_fwd_info.workspace_size = workSpaceSizeInBytes;
        tmp_if_ptr = (uintptr_t)workSpace;
        conv_fwd_info.workspace_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end() || workSpaceSizeInBytes == 0)
        {
            conv_fwd_info.if_workspace_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            conv_fwd_info.if_workspace_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&conv_fwd_info.workspace_handle, workSpace));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        tensor_out_info.tensor_name = (uintptr_t)yDesc;
        tmp_if_ptr = (uintptr_t)y;
        tensor_out_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_out_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_out_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_out_info.tensor_handle, y));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        // std::cout << "alpha " << Alpha << " beta " << Beta
        //   << " x_ptr " << (uintptr_t)x << " w_ptr " << (uintptr_t)w
        //   << " algo " << algo << " workspace_ptr " << (uintptr_t)workSpace
        //   << " workSpaceSizeInBytes " << workSpaceSizeInBytes
        //   << " y_ptr " << (uintptr_t)y << std::endl;
        std::vector<int> conv_fwd_key;
        conv_fwd_key.emplace_back(Alpha);
        conv_fwd_key.emplace_back(Beta);
        tensor_emplace(conv_fwd_key, tensor_in_info.tensor_name);
        convolution_fwd_emplace(conv_fwd_key, conv_fwd_info.filter_name, conv_fwd_info.conv_name);
        conv_fwd_key.emplace_back(algo);
        conv_fwd_key.emplace_back(workSpaceSizeInBytes);
        tensor_emplace(conv_fwd_key, tensor_out_info.tensor_name);
        compute_inter_process.send_compute(compute_type, Alpha, Beta,
                                           tensor_in_info, tensor_out_info,
                                           conv_fwd_info, activation_fwd_info,
                                           pooling_fwd_info, softmax_fwd_info,
                                           convolution_forward_dict[conv_fwd_key].bandwidth,
                                           convolution_forward_dict[conv_fwd_key].duration);
        if (compute_inter_process.compute_need())
        {
            __typeof__(cudnnConvolutionForward) *convolution_forward_fp = (__typeof__(cudnnConvolutionForward) *)dlsym(RTLD_NEXT, "cudnnConvolutionForward");
            cudnnStatus_t ret = convolution_forward_fp(handle,
                                                       alpha, xDesc, x,
                                                       wDesc, w, convDesc,
                                                       algo, workSpace, workSpaceSizeInBytes,
                                                       beta, yDesc, y);
            pthread_mutex_unlock(&api_used);
            std::cout << "Computed" << std::endl;
            return ret;
        }
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnActivationForward(cudnnHandle_t handle,
                           cudnnActivationDescriptor_t activationDesc,
                           const void *alpha,
                           const cudnnTensorDescriptor_t xDesc,
                           const void *x,
                           const void *beta,
                           const cudnnTensorDescriptor_t yDesc,
                           void *y)
    {
        std::cout << "Activation fwd hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        compute_type = ACTIVATIONFWD;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        activation_fwd_info.activation_name = (uintptr_t)activationDesc;
        tensor_in_info.tensor_name = (uintptr_t)xDesc;
        tmp_if_ptr = (uintptr_t)x;
        tensor_in_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_in_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_in_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_in_info.tensor_handle, (void *)x));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        tensor_out_info.tensor_name = (uintptr_t)yDesc;
        tmp_if_ptr = (uintptr_t)y;
        tensor_out_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_out_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_out_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_out_info.tensor_handle, y));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        // std::cout << "alpha" << Alpha << "beta" << Beta
        //   << "x_ptr" << (uintptr_t)x
        //   << "activationDesc" << (uintptr_t) activationDesc
        //   << "y_ptr" << (uintptr_t)y << std::endl;
        std::vector<int> act_fwd_key;
        activation_fwd_emplace(act_fwd_key, activation_fwd_info.activation_name);
        act_fwd_key.emplace_back(Alpha);
        act_fwd_key.emplace_back(Beta);
        tensor_emplace(act_fwd_key, tensor_in_info.tensor_name);
        tensor_emplace(act_fwd_key, tensor_out_info.tensor_name);
        compute_inter_process.send_compute(compute_type, Alpha, Beta,
                                           tensor_in_info, tensor_out_info,
                                           conv_fwd_info, activation_fwd_info,
                                           pooling_fwd_info, softmax_fwd_info,
                                           activation_forward_dict[act_fwd_key].bandwidth,
                                           activation_forward_dict[act_fwd_key].duration);
        if (compute_inter_process.compute_need())
        {
            __typeof__(cudnnActivationForward) *activation_forward_fp = (__typeof__(cudnnActivationForward) *)dlsym(RTLD_NEXT, "cudnnActivationForward");
            cudnnStatus_t ret = activation_forward_fp(handle, activationDesc,
                                                      alpha, xDesc, x,
                                                      beta, yDesc, y);
            pthread_mutex_unlock(&api_used);
            return ret;
        }
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnPoolingForward(cudnnHandle_t handle,
                        const cudnnPoolingDescriptor_t poolingDesc,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y)
    {
        std::cout << "Pooling fwd hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        compute_type = POOLFWD;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        pooling_fwd_info.pooling_name = (uintptr_t)poolingDesc;
        tensor_in_info.tensor_name = (uintptr_t)xDesc;
        tmp_if_ptr = (uintptr_t)x;
        tensor_in_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_in_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_in_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_in_info.tensor_handle, (void *)x));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        tensor_out_info.tensor_name = (uintptr_t)yDesc;
        tmp_if_ptr = (uintptr_t)y;
        tensor_out_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_out_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_out_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_out_info.tensor_handle, y));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        std::vector<int> pooling_key;
        pooling_fwd_emplace(pooling_key, pooling_fwd_info.pooling_name);
        pooling_key.emplace_back(Alpha);
        pooling_key.emplace_back(Beta);
        tensor_emplace(pooling_key, tensor_in_info.tensor_name);
        tensor_emplace(pooling_key, tensor_out_info.tensor_name);
        compute_inter_process.send_compute(compute_type, Alpha, Beta,
                                           tensor_in_info, tensor_out_info,
                                           conv_fwd_info, activation_fwd_info,
                                           pooling_fwd_info, softmax_fwd_info,
                                           pooling_forward_dict[pooling_key].bandwidth,
                                           pooling_forward_dict[pooling_key].duration);
        if (compute_inter_process.compute_need())
        {
            __typeof__(cudnnPoolingForward) *pooling_forward_fp = (__typeof__(cudnnPoolingForward) *)dlsym(RTLD_NEXT, "cudnnPoolingForward");
            cudnnStatus_t ret = pooling_forward_fp(handle, poolingDesc,
                                                   alpha, xDesc, x,
                                                   beta, yDesc, y);
            pthread_mutex_unlock(&api_used);
            return ret;
        }
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }

    cudnnStatus_t CUDNNWINAPI
    cudnnSoftmaxForward(cudnnHandle_t handle,
                        cudnnSoftmaxAlgorithm_t algo,
                        cudnnSoftmaxMode_t mode,
                        const void *alpha,
                        const cudnnTensorDescriptor_t xDesc,
                        const void *x,
                        const void *beta,
                        const cudnnTensorDescriptor_t yDesc,
                        void *y)
    {
        std::cout << "Softmax fwd hooked" << std::endl;
        pthread_mutex_lock(&api_used);
        compute_type = SOFTMAXFWD;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        softmax_fwd_info.algo = algo;
        softmax_fwd_info.mode = mode;
        tensor_in_info.tensor_name = (uintptr_t)xDesc;
        tmp_if_ptr = (uintptr_t)x;
        tensor_in_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_in_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_in_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_in_info.tensor_handle, (void *)x));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        tensor_out_info.tensor_name = (uintptr_t)yDesc;
        tmp_if_ptr = (uintptr_t)y;
        tensor_out_info.tensor_ptr = tmp_if_ptr;
        if (if_ptr_open.find(tmp_if_ptr) != if_ptr_open.end())
        {
            tensor_out_info.if_tensor_handle = false;
        }
        else
        {
            // std::cout << "opend" << std::endl;
            tensor_out_info.if_tensor_handle = true;
            CUDA_CHECK(cudaIpcGetMemHandle(&tensor_out_info.tensor_handle, y));
            if_ptr_open.emplace(tmp_if_ptr, 1);
        }
        std::vector<int> softmax_key;
        softmax_key.emplace_back(algo);
        softmax_key.emplace_back(mode);
        tensor_emplace(softmax_key, tensor_in_info.tensor_name);
        tensor_emplace(softmax_key, tensor_out_info.tensor_name);
        compute_inter_process.send_compute(compute_type, Alpha, Beta,
                                           tensor_in_info, tensor_out_info,
                                           conv_fwd_info, activation_fwd_info,
                                           pooling_fwd_info, softmax_fwd_info,
                                           softmax_forward_dict[softmax_key].bandwidth,
                                           softmax_forward_dict[softmax_key].duration);
        if (compute_inter_process.compute_need())
        {
            __typeof__(cudnnSoftmaxForward) *softmax_forward_fp = (__typeof__(cudnnSoftmaxForward) *)dlsym(RTLD_NEXT, "cudnnSoftmaxForward");
            cudnnStatus_t ret = softmax_forward_fp(handle, algo, mode,
                                                   alpha, xDesc, x,
                                                   beta, yDesc, y);
            pthread_mutex_unlock(&api_used);
            return ret;
        }
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }
}
