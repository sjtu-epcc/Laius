#include <cstdio>
#include <unistd.h>
#include <fstream>
#include <string>
#include <iostream>
#include <cuda.h>

#define cudachk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if(code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

struct LargeArg {
  char data[512];
};

__global__ void kernel(size_t param1, float *param2, LargeArg param3, int nop) {
  __shared__ float data[128];
  float local = 1234.0;
  for(size_t w = 0; w < 10240; w++) {
    for(size_t i = 0; i < 128; i += 4) {
      local += data[i];
      local /= data[i+1];
      local -= data[i+2];
      local *= data[i+3];
    }
  }
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx %= 128;
  data[idx] += local;
}

int main() {
  //cudachk( cudaMalloc(&data, 128 * sizeof(float)) );
  //int self_pid = getpid();
  //printf("%d\n",self_pid);
  //std::ifstream self_map;
  //std::string self_map_file = "/proc/" + std::to_string(self_pid) + "/maps";
  //self_map.open(self_map_file, std::ios::in);
  //std::string str_line;
  //while(getline(self_map, str_line))
  //{
    //std::cout << str_line << std::endl;
  //}
  cudaEvent_t local_start, local_stop;
  cudachk( cudaEventCreate(&local_start) );
  cudachk( cudaEventCreate(&local_stop) );
  size_t a = 1234;
  float *b;
  cudachk( cudaMalloc(&b, 4) );
  LargeArg c;
  c.data[0] = 5;
  kernel<<<128, 128>>>(a, b ,c, 0);
  //kernel<<<1024, 1024>>>();
  const int iters = 1;
  for(int i = iters; i > 0; i--) {
    cudachk( cudaEventRecord(local_start) );
    kernel<<<1024, 1024>>>(a, b, c, 0);
    cudachk( cudaEventRecord(local_stop) );
    cudachk( cudaEventSynchronize(local_stop) );
    float milliseconds = 0;
    cudachk( cudaEventElapsedTime(&milliseconds, local_start, local_stop) );
    printf("origin %fms\n", milliseconds);
  }
  //cudaFree(data);
}

