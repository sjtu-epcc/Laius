#include "kernel/ipc.h"
#include <cuda_runtime_api.h>
#include <cudnn.h>

int main()
{
    int ret;
    int fd = shm_open(getenv("SHARED_MEMORY_FNAME"),
                      O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                      S_IRUSR | S_IWUSR);
    assert(fd != -1);
    ret = ftruncate(fd, SHARED_MEM_SIZE);
    assert(ret != -1);
    SharedMemoryContents *shared_memory = (SharedMemoryContents *)mmap(NULL, SHARED_MEM_SIZE,
                                                                       PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    shared_memory->init();
    ret = close(fd);
    assert(ret != -1);
    ret = munmap(shared_memory, SHARED_MEM_SIZE);
    assert(ret != -1);
    exit(0);
}
