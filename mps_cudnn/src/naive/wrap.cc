#include <dlfcn.h>
#include <iostream>
#include <cstdint>
#include <cudnn.h>
#include <pthread.h>
#include "naive/cudnn_ipc.hpp"
struct CudnnAPI
{
    __typeof__(cudnnAddTensor) *CudnnAddTensor;
    __typeof__(cudnnConvolutionForward) *CudnnConvolutionForward;
    __typeof__(cudnnPoolingForward) *CudnnPoolingForward;
    __typeof__(cudnnActivationForward) *CudnnActivationForward;
    CudnnAPI()
    {
        CudnnAddTensor = (__typeof__(cudnnAddTensor) *)dlsym(RTLD_NEXT, "cudnnAddTensor");
        CudnnConvolutionForward = (__typeof__(cudnnConvolutionForward) *)dlsym(RTLD_NEXT, "cudnnConvolutionForward");
        CudnnPoolingForward = (__typeof__(cudnnPoolingForward) *)dlsym(RTLD_NEXT, "cudnnPoolingForward");
        CudnnActivationForward = (__typeof__(cudnnActivationForward) *)dlsym(RTLD_NEXT, "cudnnActivationForward");
    }
};

CudnnAPI cudnn_api;
static InterProcess inter_process(true);
static pthread_mutex_t api_used = PTHREAD_MUTEX_INITIALIZER;
// static int flag = 0;
API_TYPE api_type;
TensorDesc tensor_desc_in;
TensorDesc tensor_desc_out;
TensorDesc tensor_desc_etc;
ActivationDesc activation_desc;
ConvDesc conv_desc;
FilterDesc filter_desc;
PoolingDesc pooling_desc;
SoftmaxInfo softmax_info;
float Alpha, Beta;
extern "C"
{
    void getConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc, ConvDesc &conv_desc)
    {
        __typeof__(cudnnGetConvolution2dDescriptor) *fp = (__typeof__(cudnnGetConvolution2dDescriptor) *)dlsym(RTLD_DEFAULT, "cudnnGetConvolution2dDescriptor");
        fp(convDesc, &conv_desc.pad_h, &conv_desc.pad_w, &conv_desc.u, &conv_desc.v,
           &conv_desc.dilation_h, &conv_desc.dilation_w, &conv_desc.mode, &conv_desc.computeType);
    }
    void getFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc, FilterDesc &filter_desc)
    {
        __typeof__(cudnnGetFilter4dDescriptor) *fp = (__typeof__(cudnnGetFilter4dDescriptor) *)dlsym(RTLD_DEFAULT, "cudnnGetFilter4dDescriptor");
        fp(filterDesc, &filter_desc.dataType, &filter_desc.filter_format, &filter_desc.k, &filter_desc.c, &filter_desc.h, &filter_desc.w);
    }
    void getTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc, TensorDesc &tensor_desc)
    {
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
        __typeof__(cudnnGetTensor4dDescriptor) *fp = (__typeof__(cudnnGetTensor4dDescriptor) *)dlsym(RTLD_DEFAULT, "cudnnGetTensor4dDescriptor");
        fp(tensorDesc, &tensor_desc.dataType,
           &tensor_desc.n, &tensor_desc.c, &tensor_desc.h, &tensor_desc.w,
           &tensor_desc.nStride, &tensor_desc.cStride, &tensor_desc.hStride, &tensor_desc.wStride);
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
    }
    void getActivationDescriptor(const cudnnActivationDescriptor_t activationDesc, ActivationDesc &activation_desc)
    {
        __typeof__(cudnnGetActivationDescriptor) *fp = (__typeof__(cudnnGetActivationDescriptor) *)dlsym(RTLD_DEFAULT, "cudnnGetActivationDescriptor");
        fp(activationDesc, &activation_desc.mode, &activation_desc.reluNanOpt, &activation_desc.coef);
    }
    void getPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc, PoolingDesc &pooling_desc)
    {
        __typeof__(cudnnGetPooling2dDescriptor) *fp = (__typeof__(cudnnGetPooling2dDescriptor) *)dlsym(RTLD_DEFAULT, "cudnnGetPooling2dDescriptor");
        fp(poolingDesc, &pooling_desc.mode, &pooling_desc.maxpoolingNanOpt,
           &pooling_desc.windowHeight, &pooling_desc.windowWidth,
           &pooling_desc.verticalPadding, &pooling_desc.horizontalPadding,
           &pooling_desc.verticalStride, &pooling_desc.horizontalStride);
    }

    // void init()
    // {
    // std::cout << "first cudnn api" << std::endl;
    // flag = 1;
    // pthread_mutex_init(&api_used, NULL);
    // }

    cudnnStatus_t CUDNNWINAPI
    cudnnAddTensor(cudnnHandle_t handle,
                   const void *alpha,
                   const cudnnTensorDescriptor_t aDesc,
                   const void *A,
                   const void *beta,
                   const cudnnTensorDescriptor_t cDesc,
                   void *C)
    {
        // if (flag == 0)
        // init();
        // else
        // std::cout << flag << std::endl;
        pthread_mutex_lock(&api_used);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "cudnnAddTensor wrapped" << std::endl;
        api_type = ADDTENSOR_N;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_in.device_handle, (void *)A));
        // std::cout << tensor_desc_in.device_handle.reserved << std::endl;
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_out.device_handle, (void *)C));
        // std::cout << tensor_desc_out.device_handle.reserved << std::endl;
        getTensor4dDescriptor(aDesc, tensor_desc_in);
        getTensor4dDescriptor(aDesc, tensor_desc_out);
        inter_process.send_cudnn(api_type, Alpha, Beta,
                                 tensor_desc_in, tensor_desc_out, tensor_desc_etc,
                                 conv_desc, filter_desc, activation_desc, pooling_desc, softmax_info);
        // cudnnStatus_t ret = cudnn_api.CudnnAddTensor(handle,alpha,aDesc,A,beta,cDesc,C);
        // CUDNN_CHECK(ret);
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
        // return ret;
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
        // if (flag == 0)
        // init();
        // else
        // std::cout << flag << std::endl;
        pthread_mutex_lock(&api_used);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "cudnnConvolutionForward wrapped" << std::endl;
        api_type = CONVFWD_N;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        getTensor4dDescriptor(xDesc, tensor_desc_in);
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_in.device_handle, (void *)x));
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
        getFilter4dDescriptor(wDesc, filter_desc);
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
        CUDA_CHECK(cudaIpcGetMemHandle(&filter_desc.device_handle, (void *)w));
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
        getConvolution2dDescriptor(convDesc, conv_desc);
        // std::cout << getpid() << "process is at " << __LINE__ << std::endl;
        conv_desc.algo = algo;
        conv_desc.workspace_size = workSpaceSizeInBytes;
        // std::cout << workSpaceSizeInBytes << std::endl;
        // std::cout << workSpace << std::endl;
        if (workSpaceSizeInBytes != 0)
            CUDA_CHECK(cudaIpcGetMemHandle(&conv_desc.workspace_handle, workSpace));
        getTensor4dDescriptor(yDesc, tensor_desc_out);
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_out.device_handle, y));
        inter_process.send_cudnn(api_type, Alpha, Beta,
                                 tensor_desc_in, tensor_desc_out, tensor_desc_etc,
                                 conv_desc, filter_desc, activation_desc, pooling_desc, softmax_info);
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }

    // cudnnStatus_t CUDNNWINAPI
    // cudnnActivationForward(cudnnHandle_t handle,
    //                        cudnnActivationDescriptor_t activationDesc,
    //                        const void *alpha,
    //                        const cudnnTensorDescriptor_t xDesc,
    //                        const void *x,
    //                        const void *beta,
    //                        const cudnnTensorDescriptor_t yDesc,
    //                        void *y)
    // {
    //     // if (flag == 0)
    //     // init();
    //     // else
    //     // std::cout << flag << std::endl;
    //     pthread_mutex_lock(&api_used);
    //     std::cout << "cudnnActivationForward wrapped" << std::endl;
    //     api_type = ACTIVATIONFWD_N;
    //     Alpha = *(float *)alpha;
    //     Beta = *(float *)beta;
    //     getActivationDescriptor(activationDesc, activation_desc);
    //     getTensor4dDescriptor(xDesc, tensor_desc_in);
    //     CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_in.device_handle, (void *)x));
    //     getTensor4dDescriptor(yDesc, tensor_desc_out);
    //     CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_out.device_handle, y));
    //     std::cout << tensor_desc_out.device_handle.reserved << std::endl;
    //     inter_process.send_cudnn(api_type, Alpha, Beta,
    //                              tensor_desc_in, tensor_desc_out, tensor_desc_etc,
    //                              conv_desc, filter_desc, activation_desc, pooling_desc, softmax_info);
    //     pthread_mutex_unlock(&api_used);
    //     return CUDNN_STATUS_SUCCESS;
    // }

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
        // if (flag == 0)
        // init();
        // else
        // std::cout << flag << std::endl;
        pthread_mutex_lock(&api_used);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "cudnnPoolingForward wrapped" << std::endl;
        api_type = POOLFWD_N;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        getPooling2dDescriptor(poolingDesc, pooling_desc);
        getTensor4dDescriptor(xDesc, tensor_desc_in);
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_in.device_handle, (void *)x));
        getTensor4dDescriptor(yDesc, tensor_desc_out);
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_out.device_handle, y));
        inter_process.send_cudnn(api_type, Alpha, Beta,
                                 tensor_desc_in, tensor_desc_out, tensor_desc_etc,
                                 conv_desc, filter_desc, activation_desc, pooling_desc, softmax_info);
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
        // if (flag == 0)
        // init();
        // else
        // std::cout << flag << std::endl;
        pthread_mutex_lock(&api_used);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "cudnnSoftmaxForward wrapped" << std::endl;
        api_type = SOFTMAXFWD_N;
        Alpha = *(float *)alpha;
        Beta = *(float *)beta;
        softmax_info.algo = algo;
        softmax_info.mode = mode;
        getTensor4dDescriptor(xDesc, tensor_desc_in);
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_in.device_handle, (void *)x));
        getTensor4dDescriptor(yDesc, tensor_desc_out);
        CUDA_CHECK(cudaIpcGetMemHandle(&tensor_desc_out.device_handle, y));
        inter_process.send_cudnn(api_type, Alpha, Beta,
                                 tensor_desc_in, tensor_desc_out, tensor_desc_etc,
                                 conv_desc, filter_desc, activation_desc, pooling_desc, softmax_info);
        pthread_mutex_unlock(&api_used);
        return CUDNN_STATUS_SUCCESS;
    }
}
