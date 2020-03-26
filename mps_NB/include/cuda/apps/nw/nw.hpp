#pragma once
#include "cuda/apps/kernel_args.hpp"
class NwKernel : public KernelArgs
{
  private:
    void *reference;
    void *matrix_cuda;
    int cols;
    int penalty;
    int i;
    int block_width;

  public:
    NwKernel() : KernelArgs(NW_K1_PTR) {}
    void from_args(void **args_)
    {
        reference = *(void **)args_[0];
        matrix_cuda = *(void **)args_[1];
        cols = *(int *)args_[2];
        penalty = *(int *)args_[3];
        i = *(int *)args_[4];
        block_width = *(int *)args_[5];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 6);
        reference = lookup_memory_allocation(reference, memoryAllocations_, pointers_ready_, memIndex_);
        args[0] = &reference;
        matrix_cuda = lookup_memory_allocation(matrix_cuda, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &matrix_cuda;
        args[2] = (void *)&cols;
        args[3] = (void *)&penalty;
        args[4] = (void *)&i;
        args[5] = (void *)&block_width;
        return args;
    }
    int get_size()
    {
        return sizeof(NwKernel);
    }
};