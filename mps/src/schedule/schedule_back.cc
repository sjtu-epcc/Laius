#include <unordered_map>
#include <sys/time.h>
#include "cudnn/cudnn_ipc.hpp"
#include "cuda/ipc.hpp"
#include "schedule/schedule.hpp"

using namespace std;

// #define MAX_MEMORY_ALLOCATIONS 16
#define MAX_PRO_SIZE 100
#define ISSYNC 6
#define SHARED_MEM_NUM 20
float table[11];
unordered_map<uintptr_t, vector<float>> Kernel_ipc;  //all rodinia programmer kernels' ipc
unordered_map<uintptr_t, vector<float>> Kernel_band; //all rodinia programmer kernels' ipc

void schedule_with_fullPack(vector<Schedule> *kernel_schedule, vector<Schedule> Kernel_list_name, int sum_weight, int num_of_Kernel, float maxBandwidth)
{
    float **V = new float *[num_of_Kernel + 5]; //init the pack
    int Bandwidth[100][40];
    for (int i = 0; i <= num_of_Kernel; i++)
        V[i] = new float[sum_weight + 5];
    for (int i = 0; i <= num_of_Kernel; i++)
    {
        V[i][0] = 0;
        Bandwidth[i][0] = 0;
    }
    for (int i = 0; i <= sum_weight; i++)
    {
        V[0][i] = 0;
        Bandwidth[0][i] = 0;
    }
    vector<Schedule> detail[100][40];        //show the detail of the Schedule
    for (int i = 1; i <= num_of_Kernel; i++) //the full pack algorithm
    {
        for (int j = 1; j <= sum_weight; j++)
        {
            V[i][j] = V[i - 1][j];
            detail[i][j] = detail[i - 1][j];
            int maxj = j;
            if(j > 10)
                maxj = 3;
            for (int m = 1; m <= maxj; m++)
            {
                if (V[i - 1][j - m] + Kernel_ipc[Kernel_list_name[i - 1].id][m - 1] > V[i][j]) //V[i][j]=max(V[i-1][j-m]+g(i,m)) m=0,1,2,3......,10
                {
                    double tmp_band = Kernel_band[Kernel_list_name[i - 1].id][m - 1];
                    if (Bandwidth[i - 1][j] + tmp_band < maxBandwidth)
                    {
                        V[i][j] = V[i - 1][j - m] + Kernel_ipc[Kernel_list_name[i - 1].id][m - 1];
                        Bandwidth[i][j] = Bandwidth[i - 1][j];
                        detail[i][j] = detail[i - 1][j - m];
                        Schedule tmp;
                        tmp.id = Kernel_list_name[i - 1].id;
                        tmp.id = Kernel_list_name[i - 1].id;
                        tmp.GPU_ratio = m;
                        tmp.num = Kernel_list_name[i - 1].num;
                        tmp.bandwidth = tmp_band;
                        detail[i][j].push_back(tmp);
                    }
                }
            }
        }
    }
    *kernel_schedule = detail[num_of_Kernel][sum_weight];
    for (int i = 0; i < num_of_Kernel; i++)
        delete V[i];
    delete V;
    //int max = V[num_of_Kernel][sum_weight];
}
/*void schedule_with_greedy(vector<Schedule> *kernel_schedule, vector<Schedule> Kernel_list_name, int sum_weight, int num_of_Kernel)
{
    Schedule tmp;
    vector<float> delta_ipc;
    vector<float> tmp_Kernel;
    int maxdelta_ipc = 0;
    for (int i = 0; i < num_of_Kernel; i++) //init the delta_ipc
    {
        tmp.id = Kernel_list_name[i].id;
        tmp.id = Kernel_list_name[i].id;
        tmp.GPU_ratio = 0;
        (*kernel_schedule).push_back(tmp);
        delta_ipc.push_back(0);
    }
    while ((sum_weight--) > 0)
    {
        for (int i = 0; i < delta_ipc.size(); i++)
        {
            if ((*kernel_schedule)[i].GPU_ratio < 10)
            {
                tmp_Kernel = Kernel_ipc[Kernel_list_name[i].id];
                int nowWeight = (*kernel_schedule)[i].GPU_ratio;
                delta_ipc[i] = tmp_Kernel[nowWeight + 1] - tmp_Kernel[nowWeight];
            }
            else
                delta_ipc[i] = 0;
        }
        int max_delta_ipc = 0;
        int max_delta_ipc_pos = -1;
        for (int i = 0; i < delta_ipc.size(); i++)
        {
            if (delta_ipc[i] > max_delta_ipc)
            {
                max_delta_ipc = delta_ipc[i];
                max_delta_ipc_pos = i;
            }
        }
        if (max_delta_ipc_pos != -1)
            (*kernel_schedule)[max_delta_ipc_pos].GPU_ratio++;
        maxdelta_ipc += max_delta_ipc;
    }
    vector<Schedule> kernel_scheduletmp;
    for (int i = 0; i < (*kernel_schedule).size(); i++)
    {
        if ((*kernel_schedule)[i].GPU_ratio > 0)
            kernel_scheduletmp.push_back((*kernel_schedule)[i]);
    }
    *kernel_schedule = kernel_scheduletmp;
    //int max = maxdelta_ipc;
}*/
int main()
{
    printf("start scheduling\n");
    int shm_ret;
    std::string data_path = getenv("PREDICT_DATA_PATH");
    fstream file(data_path + "data_ipc.txt");
    fstream file1(data_path +  getenv("QOS_DATA"));
    fstream file2(data_path + "data_band.txt");
    vector<float> tmp;
    int tmpid = 0;
    int GPU_resource = 10;
    double max_bandwidth = 616;
    string strTmp1, strTmp2, Kernel_id;
    //read the ipc
    double time_predict = 0;
    double band_predict = 0;
    float running_time[11];
    for (int i = 0; i <= 10; i++)
        running_time[i] = 0;
    while (getline(file, strTmp1))
    {
        istringstream ss(strTmp1);
        ss >> Kernel_id;
        while (ss >> strTmp2)
            tmp.push_back(stof(strTmp2));
        Kernel_ipc[stof(Kernel_id)] = tmp;
        tmp.clear();
    }

     float a,b,c;
    int i = 1;
    while(getline(file1,strTmp1)){
        istringstream ss(strTmp1);
        ss>>a>>b;
        table[i++] = b;
    }
    int minTime = 0;
    double minTT = 10000000000;
    for(int i=1;i<=10;i++)
    {
        if(table[i] < minTT)
        {
            minTT = table[i];
            minTime = i;
        }
    }
    printf("%d\n",minTime);
    double timeuse = 0;
    while (getline(file2, strTmp1))
    {
        istringstream ss(strTmp1);
        ss >> Kernel_id;
        while (ss >> strTmp2)
            tmp.push_back(stof(strTmp2));
        Kernel_band[stof(Kernel_id)] = tmp;
        tmp.clear();
    }
    //int a;
    ShmCompute *cudnn_api;
    int fd = shm_open(getenv("SHARED_COMPUTE_FNAME"), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    assert(fd != -1);
    cudnn_api = (ShmCompute *)mmap(NULL, sizeof(ShmCompute), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(close(fd) != -1);
    vector<string> processName = getProcessName();
    SharedMemoryContents *pKernel_list_name[SHARED_MEM_NUM]; // = new SharedMemoryContents* [SHARED_MEM_NUM];
    for (int i = 0; i < processName.size(); i++)
    {
        int fd = shm_open(processName[i].c_str(),
                          O_RDWR | O_CREAT | O_TRUNC | O_EXCL,
                          S_IRUSR | S_IWUSR);
        assert(fd != -1);
        int ret = ftruncate(fd, SHARED_MEM_SIZE);
        pKernel_list_name[i] = (SharedMemoryContents *)mmap(NULL, SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        pKernel_list_name[i]->init();
        assert(close(fd) != -1);
    }
    printf("22222\n");
    int processNum = processName.size();
    printf("%d\n", processNum);
    int isLate = 0;
    //timeval starttime,endtime;
    int counter = 0;
    timeval startapi, endapi, tmpapi;
    vector<Schedule> *kernel_schedule; //Kernel's Schedule
    kernel_schedule = new vector<Schedule>;
    int list_size = processNum;
    vector<Schedule> *need_schedule = new vector<Schedule>;
    int predict = 0;
    double  sum_band;
    while (true)
    {
        //printf("please input something\n");
        //scanf("%d",&a);
        //printf("using %d percentage\n",cudnn_api->percent_flag);
        if (cudnn_api->process_flag == 0)
        {
            printf("the process_flag is %d and changing to 1\n", cudnn_api->process_flag);
            for (int i = 1; i <= 10; i++)
            {
                running_time[i] += cudnn_api->cudnn_call.duration[i];
            }
            time_predict += cudnn_api->cudnn_call.duration[cudnn_api->self_percent / 10];
           /* for (int i=1;i<=10;i++)
            {
                printf("%lf ", cudnn_api->cudnn_call.duration[i]);
            }
            printf("\n");*/
            counter++;
            if (counter % ISSYNC == 0)
            {
                cudnn_api->if_sync = 1;
            }
            else if (counter % ISSYNC == 1)
            {
                cudnn_api->if_sync = 0;
                if (isLate == 1)
                {
                    // ......
                    int tmp_ratio = cudnn_api->percent_flag;
                    printf("%d %d\n",tmp_ratio,minTime);
                    for (int i = cudnn_api->percent_flag; i <= 70; i += 10)
                    {
                        cudnn_api->percent_flag = i;
                        if ((table[tmp_ratio / 10] - table[i / 10]) -
                                (running_time[tmp_ratio / 10] - running_time[i / 10]) >=
                            timeuse - time_predict){
                            break;}
                    }
                    printf("using %d percentage\n",cudnn_api->percent_flag);
                    isLate = 0;
                }

                GPU_resource -= cudnn_api->percent_flag / 10;
                gettimeofday(&startapi, 0);

            }

            band_predict = cudnn_api->cudnn_call.bandwidth[cudnn_api->percent_flag / 10];
            max_bandwidth -= band_predict;
            sum_band += band_predict;
            cudnn_api->process_flag = 1;
            if (cudnn_api->self_percent == cudnn_api->percent_flag)
            {
               // printf("cc\n");
                pthread_barrier_wait(&cudnn_api->sync.barrier); //不切换
                pthread_barrier_wait(&cudnn_api->sync.barrier);
               // printf("the process flag is %d\n", cudnn_api->process_flag);
                // if(cudnn_api->if_sync == 1){
                //     cudnn_api->process_flag = 2;
                // }
                // else
                //     cudnn_api->process_flag = -1;
               // printf("dd\n");
            }

            //cudnn_api->process_flag = -1;
        }
        if (cudnn_api->process_flag == 2)
        {
            GPU_resource += cudnn_api->percent_flag / 10;
            max_bandwidth += sum_band;
            cudnn_api->process_flag = 3;
            gettimeofday(&endapi, 0);
            timeuse = 1000000 * (endapi.tv_sec - startapi.tv_sec) + endapi.tv_usec - startapi.tv_usec;
          //  timeuse /= 1000;
            if(timeuse >= 221000)
                timeuse -= 221000;
            printf("%lf\n",timeuse);
            printf("%lf\n",time_predict);
            if (timeuse > time_predict*1.3 && counter / ISSYNC != 1)//try
            {
                printf("totally late\n");
                isLate = 1;
            }
            time_predict = 0;
            predict = time_predict;
        }
        for (int i = 0; i < list_size; i++)
        {
            if (pKernel_list_name[i]->sch.process_flag == 0) //&& pKernel_list_name[i].sch.change_flag == 1)
            {
           //     printf("the process_flag is %d\n",pKernel_list_name[i]->sch.process_flag);
                pKernel_list_name[i]->sch.num = i;
                (*need_schedule).push_back(pKernel_list_name[i]->sch);
            }
            if (pKernel_list_name[i]->sch.process_flag == 2)
            {
               // printf("the process_flag is %d\n",pKernel_list_name[i]->sch.process_flag);
                GPU_resource += pKernel_list_name[i]->sch.GPU_ratio;
                max_bandwidth += pKernel_list_name[i]->sch.bandwidth;
            //    printf("GPU_ratio get %d back from %d\n", pKernel_list_name[i]->sch.GPU_ratio, i);
                pKernel_list_name[i]->sch.GPU_ratio = 0;
                pKernel_list_name[i]->sch.process_flag = -1;
               // printf("the process_flag is %d\n",pKernel_list_name[i]->sch.process_flag);
            }
        }
        //choose one
        //schedule_with_greedy(kernel_schedule, *need_schedule, 10, (*need_schedule).size());
        if (GPU_resource > 0 && (*need_schedule).size() > 0)
        {
           // printf("sum_kernel is %d\n", (*need_schedule).size());
          //  printf("the GPU_ratio is %d\n", GPU_resource);
            schedule_with_fullPack(kernel_schedule, *need_schedule, GPU_resource, (*need_schedule).size(), max_bandwidth);
        }
        for (int i = 0; i < (*kernel_schedule).size(); i++)
        {
            int Kernel_pos = (*kernel_schedule)[i].num;
          //  printf("this is the %d process\n",Kernel_pos);
            //          if(pKernel_list_name[Kernel_pos]->sch.GPU_ratio>0)
            //          {
            pKernel_list_name[Kernel_pos]->sch = (*kernel_schedule)[i];
            if(pKernel_list_name[Kernel_pos]->sch.GPU_ratio>8)
            {
                pKernel_list_name[Kernel_pos]->sch.GPU_ratio = 8;
            }
            GPU_resource -= pKernel_list_name[Kernel_pos]->sch.GPU_ratio;
            max_bandwidth -= pKernel_list_name[Kernel_pos]->sch.bandwidth;
            pKernel_list_name[Kernel_pos]->sch.process_flag = 1;
           // printf("process[%d] get %d GPU\n", Kernel_pos, pKernel_list_name[Kernel_pos]->sch.GPU_ratio);
            //          }
        }
        kernel_schedule->clear();
        need_schedule->clear();
    }
    delete kernel_schedule;
    delete need_schedule;
    delete pKernel_list_name;
}
