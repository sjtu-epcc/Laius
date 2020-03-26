#include <dlfcn.h>
#include <iostream>
#include <cstdint>
#include <cudnn.h>
#include <pthread.h>

static pthread_mutex_t api_used = PTHREAD_MUTEX_INITIALIZER;
// static int flag = 0;
cudnnStatus_t CUDNNWINAPI
cudnnAddTensor(cudnnHandle_t handle,
               const void *alpha,
               const cudnnTensorDescriptor_t aDesc,
               const void *A,
               const void *beta,
               const cudnnTensorDescriptor_t cDesc,
               void *C)
{
    pthread_mutex_lock(&api_used);
    __typeof__(cudnnAddTensor) *CudnnAddTensor = (__typeof__(cudnnAddTensor) *)dlsym(RTLD_NEXT, "cudnnAddTensor");
    CudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
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
    pthread_mutex_lock(&api_used);
    __typeof__(cudnnConvolutionForward) *CudnnConvolutionForward = (__typeof__(cudnnConvolutionForward) *)dlsym(RTLD_NEXT, "cudnnConvolutionForward");
    CudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
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
    pthread_mutex_lock(&api_used);
    __typeof__(cudnnActivationForward) *CudnnActivationForward = (__typeof__(cudnnActivationForward) *)dlsym(RTLD_NEXT, "cudnnActivationForward");
    CudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
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
    pthread_mutex_lock(&api_used);
    __typeof__(cudnnPoolingForward) *CudnnPoolingForward = (__typeof__(cudnnPoolingForward) *)dlsym(RTLD_NEXT, "cudnnPoolingForward");
    CudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
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
    pthread_mutex_lock(&api_used);
    __typeof__(cudnnSoftmaxForward) *CudnnSoftmaxForward = (__typeof__(cudnnSoftmaxForward) *)dlsym(RTLD_NEXT, "cudnnSoftmaxForward");
    CudnnSoftmaxForward(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
    pthread_mutex_unlock(&api_used);
    return CUDNN_STATUS_SUCCESS;
}
