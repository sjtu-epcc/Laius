#include "naive/cudnn_ipc.hpp"

int main(int argc, char const *argv[])
{
    int ret;
    int fd = shm_open(getenv("SHARED_MEMORY_FNAME"),
                      O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                      S_IRUSR | S_IWUSR);
    assert(fd != -1);
    // fprintf(stderr, "process %d is at line %d\n", getpid(), __LINE__);
    ret = ftruncate(fd, SHARED_SIZE);
    struct stat shm_status;
    fstat(fd, &shm_status);
    std::cout << shm_status.st_size << std::endl;
    std::cout << SHARED_SIZE << " " << SHARED_MEM_SIZE << " " << SHARED_MEM_NUM << " " << SHM_FLAG_SIZE << std::endl;
    assert(ret != -1);
    void *init_ptr = mmap(NULL, SHARED_SIZE,
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    uintptr_t init_addr = (uintptr_t)init_ptr;
    ShmUseFlag *shm_use_flag = (ShmUseFlag *)init_addr;
    shm_use_flag->init();

    // SharedMemoryContents *shared_memory = (SharedMemoryContents *)mmap(NULL, SHARED_MEM_SIZE,
    //  PROT_READ | PROT_WRITE,
    //  MAP_SHARED, fd,
    //  0);
    // shared_memory->init();
    // ret = munmap(shared_memory, SHARED_MEM_SIZE);
    SharedMemoryContents *shared_memory;
    for (int i = 0; i < SHARED_MEM_NUM; i++)
    {
        shared_memory = (SharedMemoryContents *)(init_addr + SHM_FLAG_SIZE + SHARED_MEM_SIZE * i);
        shared_memory->init();
    }
    ret = close(fd);
    assert(ret != -1);
    ret = munmap(init_ptr, SHARED_SIZE);
    assert(ret != -1);
    return 0;
}
