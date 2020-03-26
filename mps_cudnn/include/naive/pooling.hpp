#include "naive/base_api.hpp"

class pooling : base_api
{
  public:
    cudnnPoolingDescriptor_t tensor_pooling;
    pooling()
    {
        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&tensor_pooling));
    }
    void setup(SharedMemoryContents *shared_memory)
    {
        base_setup(shared_memory);
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(tensor_pooling,
                                                shared_memory->cudnn_call.pooling_desc.mode,
                                                shared_memory->cudnn_call.pooling_desc.maxpoolingNanOpt,
                                                shared_memory->cudnn_call.pooling_desc.windowHeight,
                                                shared_memory->cudnn_call.pooling_desc.windowWidth,
                                                shared_memory->cudnn_call.pooling_desc.verticalPadding,
                                                shared_memory->cudnn_call.pooling_desc.horizontalPadding,
                                                shared_memory->cudnn_call.pooling_desc.verticalStride,
                                                shared_memory->cudnn_call.pooling_desc.horizontalStride));
    }
    void compute(cudnnHandle_t &cudnn_handle,cudaEvent_t &event_, SharedMemoryContents *shared_memory)
    {
        CUDNN_CHECK(cudnnPoolingForward(cudnn_handle, tensor_pooling, &alpha,
                                        tensor_in, tensor_in_ptr, &beta,
                                        tensor_out, tensor_out_ptr));
        // if (shared_memory->change_flag == 1)
        // {
            // synchronize(shared_memory);
        // }
        shared_memory->process_flag = -1;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_in_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaIpcCloseMemHandle(tensor_out_ptr));
        // std::cout << tensor_out_ptr << std::endl;
        CUDA_CHECK(cudaEventRecord(event_));
        // CUDA_CHECK(cudaDeviceSynchronize());
        pthread_barrier_wait(&shared_memory->sync.barrier);
    }
};