#pragma once
#include "cuda/apps/kernel_args.hpp"

class BplustreeKernel1 : public KernelArgs
{
  private:
    long height;
    void *knodesD;
    long knodes_elem;
    void *recordsD;
    void *currKnodeD;
    void *offsetD;
    void *keysD;
    void *ansD;

  public:
    BplustreeKernel1() : KernelArgs(BPLUSTREE_K1_PTR) {}
    void from_args(void **args_)
    {
        height = *(long *)args_[0];
        knodesD = *(void **)args_[1];
        knodes_elem = *(long *)args_[2];
        recordsD = *(void **)args_[3];
        currKnodeD = *(void **)args_[4];
        offsetD = *(void **)args_[5];
        keysD = *(void **)args_[6];
        ansD = *(void **)args_[7];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 8);
        args[0] = (void *)&height;
        knodesD = lookup_memory_allocation(knodesD, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &knodesD;
        args[2] = (void *)&knodes_elem;
        recordsD = lookup_memory_allocation(recordsD, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &recordsD;
        currKnodeD = lookup_memory_allocation(currKnodeD, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &currKnodeD;
        offsetD = lookup_memory_allocation(offsetD, memoryAllocations_, pointers_ready_, memIndex_);
        args[5] = &offsetD;
        keysD = lookup_memory_allocation(keysD, memoryAllocations_, pointers_ready_, memIndex_);
        args[6] = &keysD;
        ansD = lookup_memory_allocation(ansD, memoryAllocations_, pointers_ready_, memIndex_);
        args[7] = &ansD;
        return args;
    }
    int get_size()
    {
        return sizeof(BplustreeKernel1);
    }
};

class BplustreeKernel2 : public KernelArgs
{
  private:
    long height;
    void *knodesD;
    long knodes_elem;
    void *currKnodeD;
    void *offsetD;
    void *lastKnodeD;
    void *offset_2D;
    void *startD;
    void *endD;
    void *RecstartD;
    void *ReclenD;

  public:
    BplustreeKernel2() : KernelArgs(BPLUSTREE_K2_PTR) {}
    void from_args(void **args_)
    {
        height = *(long *)args_[0];
        knodesD = *(void **)args_[1];
        knodes_elem = *(long *)args_[2];
        currKnodeD = *(void **)args_[3];
        offsetD = *(void **)args_[4];
        lastKnodeD = *(void **)args_[5];
        offset_2D = *(void **)args_[6];
        startD = *(void **)args_[7];
        endD = *(void **)args_[8];
        RecstartD = *(void **)args_[9];
        ReclenD = *(void **)args_[10];
    }
    void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                   std::unordered_map<void *, void *> &pointers_ready_,
                   int memIndex_)
    {
        void **args = (void **)malloc(sizeof(void *) * 11);
        args[0] = (void *)&height;
        knodesD = lookup_memory_allocation(knodesD, memoryAllocations_, pointers_ready_, memIndex_);
        args[1] = &knodesD;
        args[2] = (void *)&knodes_elem;
        currKnodeD = lookup_memory_allocation(currKnodeD, memoryAllocations_, pointers_ready_, memIndex_);
        args[3] = &currKnodeD;
        offsetD = lookup_memory_allocation(offsetD, memoryAllocations_, pointers_ready_, memIndex_);
        args[4] = &offsetD;
        lastKnodeD = lookup_memory_allocation(lastKnodeD, memoryAllocations_, pointers_ready_, memIndex_);
        args[5] = &lastKnodeD;
        offset_2D = lookup_memory_allocation(offset_2D, memoryAllocations_, pointers_ready_, memIndex_);
        args[6] = &offset_2D;
        startD = lookup_memory_allocation(startD, memoryAllocations_, pointers_ready_, memIndex_);
        args[7] = &startD;
        endD = lookup_memory_allocation(endD, memoryAllocations_, pointers_ready_, memIndex_);
        args[8] = &endD;
        RecstartD = lookup_memory_allocation(RecstartD, memoryAllocations_, pointers_ready_, memIndex_);
        args[9] = &RecstartD;
        ReclenD = lookup_memory_allocation(ReclenD, memoryAllocations_, pointers_ready_, memIndex_);
        args[10] = &ReclenD;
        return args;
    }
    int get_size()
    {
        return sizeof(BplustreeKernel2);
    }
};