#pragma once
#include "cuda/apps/kernel_args.hpp"


class LudKernel : public KernelArgs
{
  private:
    void *m;
    int matrix_dim;
    int offset;

  public:
    LudKernel() : KernelArgs(1234) {}
    void from_args(void **args_)
    {
        m = *(void **)args_[0];
        matrix_dim = *(int *)args_[1];
        offset = *(int *)args_[2];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 3);
        m = lookup_memory_allocation(m, memoryAllocations_, pointers_ready_, memIndex_);
        args[0] = &m;
        args[1] = (void *)&matrix_dim;
        args[2] = (void *)&offset;
        return args;
    }
    int get_size()
    {
        return sizeof(LudKernel);
    }
};
