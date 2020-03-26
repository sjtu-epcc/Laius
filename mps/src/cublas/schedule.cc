#include <unordered_map>
#include <sys/time.h>
#include <cublas/cublas_ipc.hpp>
#include "cuda/ipc.hpp"
#include "schedule/schedule.hpp"
#include <semaphore.h>
int main()
{

    int fd = shm_open(getenv("SHARED_CUBLAS_FNAME"),
                  O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                  S_IRUSR | S_IWUSR);
    CHECK_NE(fd, -1);
    int ret = ftruncate(fd, SHM_CUBLAS_SIZE);
    CHECK_NE(ret, -1);
    DLOG(INFO) << "Shared Memory for CUBLAS API Created Successfully!";
    ShmCublas *shm_cublas = (ShmCublas *)mmap(NULL, SHM_CUBLAS_SIZE,
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED, fd, 0);
    shm_cublas->init();
    while (true)
    {
        if (sem_trywait(&shm_cublas->sync.sch_sem) == 0)
        {
            sem_post(&shm_cublas->sync.if_sync);
            sem_post(&shm_cublas->sync.percent_sem[10]);
            // sem_post(&shm_cublas->sync.percent_sem[0]);
            // pthread_barrier_wait(&shm_cublas->sync.barrier); //不切换
            // sem_post(&cudnn_api->sync.percent_sem[cudnn_api->percent_flag / 10]);
        }

        //cudnn_api->process_flag = -1;

        if (sem_trywait(&shm_cublas->sync.synced) == 0)
        {
        }
    }
}
