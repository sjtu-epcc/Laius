#pragma once
#include "naive/base_api.hpp"
class addtensor : base_api
{
//   private:
    // float alpha, beta;
    // cudnnTensorDescriptor_t tensor_in;
    // cudnnTensorDescriptor_t tensor_out;
    // void *tensor_in_ptr;
    // void *tensor_out_ptr;

  public:
    // addtensor()
    // {
    //     CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_in));
    //     CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_out));
    // }
    void setup(SharedMemoryContents *shared_memory)
    {
        base_setup(shared_memory);
    }
    void compute(cudnnHandle_t &cudnn_handle,cudaEvent_t &event_, SharedMemoryContents *shared_memory)
    {
        CUDNN_CHECK(cudnnAddTensor(cudnn_handle,
                                   &alpha, tensor_in, tensor_in_ptr,
                                   &beta, tensor_out, tensor_out_ptr));
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
