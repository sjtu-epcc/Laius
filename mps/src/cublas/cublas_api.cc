#include "cublas/cublas_api.hpp"

void cublas_api_server(cublasHandle_t *cublas_handle,
                       ShmCublas *shm_cublas,
                       DeviceStore *dev_store,
                       int &cur_percent_, int &cur_pid_)
{
    Sgemm sgemm_sovler;
    int compute_cnt = 0;
    int compute_total = 0;
    DLOG(INFO) << "Wait for Cublas API!";
    while (true)
    {
        // int sem_val;
        // sem_getvalue(&shm_cublas->sync.sch_sem, &sem_val);
        // DLOG(INFO) << "------CURRENT-STAGE-" << sem_val << "------";
        if (sem_trywait(&shm_cublas->sync.percent_sem[cur_percent_]) == 0)
        // if (sem_trywait(&shm_cublas->sync.sch_sem) == 0)
        {
            compute_cnt++;
            compute_total++;

            DLOG(INFO) << "------" << compute_total << "-th api computed------"
                       << " Using " << cur_percent_ * 10 << " percentage";
            // shm_compute->if_computed = true;
            // DLOG(INFO) << shm_compute[i]->cudnn_call.api_type<< std::endl;
            switch (shm_cublas->cublas_call.api_type)
            {
            case SGEMM:
                DLOG(INFO) << "Sgemm received";
                sgemm_sovler.set_up(shm_cublas, dev_store);
                sgemm_sovler.compute(cublas_handle);
                break;
            }
            if (sem_trywait(&shm_cublas->sync.if_sync) == 0)
            {
                CUDA_CHECK(cudaDeviceSynchronize());
                sem_post(&shm_cublas->sync.synced);
                DLOG(INFO) << "------synced------";
            }
            pthread_barrier_wait(&shm_cublas->sync.barrier);
        }
    }
}