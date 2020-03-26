#pragma once
#include "cuda/apps/kernel_args.hpp"

class PathfinderKernel : public KernelArgs
{
  private:
    int iteration;
    void *gpuWall;
    void *gpuSRC;
    void *gpuResults;
    int cols;
    int rows;
    int startStep;
    int border;

  public:
    PathfinderKernel() : KernelArgs(PATHFINDER_K_PTR) {}
    void from_args(void **args_)
    {
        iteration = *(int *)args_[0];
        gpuWall = *(void **)args_[1];
        gpuSRC = *(void **)args_[2];
        gpuResults = *(void **)args_[3];
        cols = *(int *)args_[4];
        rows = *(int *)args_[5];
        startStep = *(int *)args_[6];
        border = *(int *)args_[7];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 8);
        args[0] = (void *)&iteration;
        gpuWall = lookup_memory_allocation(gpuWall, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &gpuWall;
        gpuSRC = lookup_memory_allocation(gpuSRC, memoryAllocations_, pointers_ready_, memIndex_);
        args[2] = &gpuSRC;
        gpuResults = lookup_memory_allocation(gpuResults, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &gpuResults;
        args[4] = (void *)&cols;
        args[5] = (void *)&rows;
        args[6] = (void *)&startStep;
        args[7] = (void *)&border;
        return args;
    }
    int get_size()
    {
        return sizeof(PathfinderKernel);
    }
};