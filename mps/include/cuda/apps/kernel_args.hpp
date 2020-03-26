#pragma once
#include "cuda/common.hpp"

#define BFS_K1_PTR 4206065
#define BFS_K2_PTR 4206570
#define BPLUSTREE_K1_PTR 4217200
#define BPLUSTREE_K2_PTR 4220355
#define HOTSOT_K_PTR 4205895
#define KMEANS_K1_PTR 4201984
#define KMEANS_K2_PTR 4202992
#define LAVAMD_K_PTR 4202615
#define LUD_K1_PTR 4209526
#define LUD_K2_PTR 4209926
#define LUD_K3_PTR 4210326
#define MYOCYTE_K1_PTR 4214720
#define MYOCYTE_K2_PTR 4215496
#define NW_K1_PTR 4205935
#define NW_K2_PTR 4206467
#define PATHFINDER_K_PTR 4203931

void *lookup_memory_allocation(void *ptr,
                               std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                               std::unordered_map<void *, void *> &pointers_ready_,
                               int memIndex_)
{
    std::unordered_map<void *, void *>::iterator it_pointer = pointers_ready_.find(ptr);
    if (it_pointer != pointers_ready_.end())
        return it_pointer->second;
    for (int i = 0; i < memIndex_; i++)
    {
        if (std::get<0>(memoryAllocations_[i]) == ptr)
        {
            void *local_ptr;
            cudachk(cudaIpcOpenMemHandle(&local_ptr, std::get<1>(memoryAllocations_[i]),
                                         cudaIpcMemLazyEnablePeerAccess));
            pointers_ready_.emplace(ptr, local_ptr);
            return local_ptr;
        }
    }
    std::abort();
}

class KernelArgs
{
  public:
    std::uintptr_t kernel_ptr;
    KernelArgs(std::uintptr_t kernel_ptr_) : kernel_ptr(kernel_ptr_) {}
    virtual void from_args(void **args_) = 0;
    virtual void **to_args(std::tuple<void *, cudaIpcMemHandle_t> memoryAllocations_[],
                           std::unordered_map<void *, void *> &pointers_ready_,
                           int memIndex_) = 0;
    virtual int get_size() = 0;
};
