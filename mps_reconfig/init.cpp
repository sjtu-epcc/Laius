#include "ipc.hpp"
#include "cudnn_ipc.hpp"
int main() {
  vector<string> processName = getProcessName();
  SharedMemoryContents **shared_memory = new SharedMemoryContents* [SHARED_MEM_NUM]; 
  ShmCompute *cudnn_api;
  printf("00000\n");
  int fd = shm_open("shared_compute", O_RDWR|O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR); 
  ftruncate(fd,sizeof(ShmCompute));
  printf("11111\n");
  cudnn_api = (ShmCompute*)mmap(NULL,sizeof(ShmCompute),PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  cudnn_api->init();
  close(fd);
  int ret;
  for(int i=0;i<processName.size();i++)
  {	  
  	int fd = shm_open(processName[i].c_str(), O_RDWR|O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR);
  	assert(fd != -1);
  	ret = ftruncate(fd, SHARED_MEM_SIZE);
  	assert(ret != -1);
   	shared_memory[i] = (SharedMemoryContents*)mmap(NULL, SHARED_MEM_SIZE,PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  	ret = close(fd);
  	assert(ret != -1);
	shared_memory[i]->init();
  	ret = munmap(shared_memory[i], SHARED_MEM_SIZE);
	assert(ret != -1);
  }
  delete shared_memory;
  exit(0);
}
