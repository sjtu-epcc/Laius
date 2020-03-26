#pragma once
#include "cuda/apps/kernel_args.hpp"
#include "cuda/apps/bfs/bfs.hpp"
#include "cuda/apps/bplustree/bplustree.hpp"
#include "cuda/apps/hotspot/hotspot.hpp"
#include "cuda/apps/kmeans/kmeans.hpp"
#include "cuda/apps/lavaMD/lavaMD.hpp"
#include "cuda/apps/lud/lud.hpp"
#include "cuda/apps/myocyte/myocyte.hpp"
#include "cuda/apps/nw/nw.hpp"
#include "cuda/apps/pathfinder/pathfinder.hpp"
//bfs
#define BFS_K1_SIZE sizeof(BfsKernel1)
#define BFS_K2_SIZE sizeof(BfsKernel2)
//bplustree
#define BPLUSTREE_K1_SIZE sizeof(BplustreeKernel1)
#define BPLUSTREE_K2_SIZE sizeof(BplustreeKernel2)
//hotspot
#define HOTSOT_K_SIZE sizeof(HotspotKernel)
//kmeans
#define KMEANS_K1_SIZE sizeof(KmeansKernel1)
#define KMEANS_K2_SIZE sizeof(KmeansKernel2)
//lavaMD
#define LAVAMD_K_SIZE sizeof(LavaMDKernel)
//lud
#define LUD_K_SIZE sizeof(LudKernel)
//myoctye
#define MYOCTYE_K1_SIZE sizeof(MyocyteKernel1)
#define MYOCTYE_K2_SIZE sizeof(MyocyteKernel2)
//nw
#define NW_K_SIZE sizeof(NwKernel)
//pathfinder
#define PATHFINDER_K_SIZE sizeof(PathfinderKernel)
//total
#define K_SIZE 120
