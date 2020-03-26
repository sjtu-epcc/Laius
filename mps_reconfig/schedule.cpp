#define MAX_PRO_SIZE 100
#include <map>
#include "ipc.h"
using namespace std;
#define MAX_MEMORY_ALLOCATIONS 16
#include <sys/time.h>

map<string, vector<float>> Kernel_ipc;//all rodinia programmer kernels' ipc
void schedule_with_fullPack(vector<schedule> *kernel_schedule, vector<schedule> Kernel_list_name, int sum_weight, int num_of_Kernel)
{
	float **V = new float *[num_of_Kernel + 5];//init the pack
	for (int i = 0; i <= num_of_Kernel; i++)
		V[i] = new float[sum_weight + 5];
	for (int i = 0; i <= num_of_Kernel; i++)
		V[i][0] = 0;
	for (int i = 0; i <= sum_weight; i++)
		V[0][i] = 0;
	vector<schedule> detail[100][40];//show the detail of the schedule
	for (int i = 1; i <= num_of_Kernel; i++)//the full pack algorithm
	{
		for (int j = 1; j <= sum_weight; j++)
		{
			V[i][j] = V[i - 1][j];
			detail[i][j] = detail[i - 1][j];
			for (int m = 1; m <= j; m++)
				if (V[i - 1][j - m] + Kernel_ipc[Kernel_list_name[i - 1].kernel_name][m - 1] > V[i][j])//V[i][j]=max(V[i-1][j-m]+g(i,m)) m=0,1,2,3......,10
				{
					V[i][j] = V[i - 1][j - m] + Kernel_ipc[Kernel_list_name[i - 1].kernel_name][m - 1];
					detail[i][j] = detail[i - 1][j - m];
					schedule tmp;
					tmp.kernel_name = Kernel_list_name[i - 1].kernel_name;
					tmp.id = Kernel_list_name[i - 1].id;
					tmp.GPU_ratio = m;
					detail[i][j].push_back(tmp);
				}
		}
	}
	*kernel_schedule = detail[num_of_Kernel][sum_weight];
	for(int i=0;i<num_of_Kernel;i++)
		delete V[i];
	delete V;
	//int max = V[num_of_Kernel][sum_weight];
}
void schedule_with_greedy(vector<schedule> *kernel_schedule, vector<schedule> Kernel_list_name, int sum_weight, int num_of_Kernel)
{
	schedule tmp;
	vector<float> delta_ipc;
	vector<float> tmp_Kernel;
	int maxdelta_ipc = 0;
	for (int i = 0; i < num_of_Kernel; i++)//init the delta_ipc
	{
		tmp.kernel_name = Kernel_list_name[i].kernel_name;
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
				tmp_Kernel = Kernel_ipc[Kernel_list_name[i].kernel_name];
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
	vector<schedule> kernel_scheduletmp;
	for (int i = 0; i < (*kernel_schedule).size(); i++)
	{
		if ((*kernel_schedule)[i].GPU_ratio > 0)
			kernel_scheduletmp.push_back((*kernel_schedule)[i]);
	}
	*kernel_schedule = kernel_scheduletmp;
	//int max = maxdelta_ipc;
}
int main()
{
	int shm_ret;
	fstream file("data_ipc.txt");
	vector<float> tmp;
	int tmpid = 0;
	int GPU_resource = 10;
	string strTmp1, strTmp2, Kernel_name;
	//read the ipc
	while (getline(file, strTmp1))
	{
		istringstream ss(strTmp1);
		ss >> Kernel_name;
		while (ss >> strTmp2)
			tmp.push_back(stof(strTmp2));
		Kernel_ipc[Kernel_name] = tmp;
		tmp.clear();
	}
	//int a;
	vector<string> processName = getProcessName();
	SharedMemoryContents **pKernel_list_name = new SharedMemoryContents* [SHARED_MEM_NUM];
	for(int i=0;i<processName.size();i++)
	{
		int fd = shm_open(processName[i].c_str(), O_RDWR|O_CREAT, S_IRUSR | S_IWUSR);
		assert(fd != -1);
		void *init_ptr = mmap(NULL, SHARED_MEM_SIZE,PROT_READ | PROT_WRITE,MAP_SHARED, fd, 0);
		pKernel_list_name[i] = (SharedMemoryContents *)init_ptr;
		assert(close(fd) != -1);
	}
	int processNum = processName.size();
	printf("%d\n",processNum);
	int busy[2]={0,0};
	timeval starttime,endtime;
	while (true)
	{
		//printf("please input something\n");
		//scanf("%d",&a);
		vector<schedule> *kernel_schedule;//Kernel's schedule
		kernel_schedule = new vector<schedule>;
		int list_size = processNum;
		vector<schedule> *need_schedule = new vector<schedule>;
		map<int, int> Kernel_position;
		for (int i = 0; i < list_size; i++)
		{
			if(pKernel_list_name[i]->sch.fptr==7360)
				pKernel_list_name[i]->sch.kernel_name = "Kernel2(bool*,bool*,bool*,bool*,int)";
			else
				pKernel_list_name[i]->sch.kernel_name = "Kernel(Node*,int*,bool*,bool*,bool*,int*,int)";
			//iprintf("%d ",pKernel_list_name[i]->sch.process_flag);
			if (pKernel_list_name[i]->sch.process_flag == 0 )//&& pKernel_list_name[i].sch.change_flag == 1)
			{
				pKernel_list_name[i]->sch.id = i;
				(*need_schedule).push_back(pKernel_list_name[i]->sch);
				Kernel_position[pKernel_list_name[i]->sch.id] = i;
				busy[i]=1;
			}
			if (pKernel_list_name[i]->sch.process_flag == 2 )
			{
				pKernel_list_name[i]->sch.process_flag = -1;
				GPU_resource += pKernel_list_name[i]->sch.GPU_ratio;
				printf("GPU_ratio get %d back from %d\n",pKernel_list_name[i]->sch.GPU_ratio,i);
			}
		}
		//choose one
		//schedule_with_greedy(kernel_schedule, *need_schedule, 10, (*need_schedule).size());
		if(GPU_resource>0 && (*need_schedule).size()>0)
		{
//			gettimeofday(&starttime,0);
			printf("\n\n");
			printf("sum_kernel is %d\n",(*need_schedule).size());
			printf("the GPU_ratio is %d\n",GPU_resource);
			schedule_with_fullPack(kernel_schedule, *need_schedule, GPU_resource, (*need_schedule).size());
//			gettimeofday(&endtime,0);
//			double timeuse  = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
//			timeuse /=1000;
//			printf("schedule used %lf\n",timeuse);
		}
		for (int i = 0; i < (*kernel_schedule).size(); i++)
		{
			int Kernel_pos = (*kernel_schedule)[i].id;
//			if(pKernel_list_name[Kernel_pos]->sch.GPU_ratio>0)
//			{
			pKernel_list_name[Kernel_pos]->sch = (*kernel_schedule)[i];
			GPU_resource -= pKernel_list_name[Kernel_pos]->sch.GPU_ratio;
			pKernel_list_name[Kernel_pos]->sch.process_flag = 1;
			printf("process[%d] get %d GPU\n",Kernel_pos,pKernel_list_name[Kernel_pos]->sch.GPU_ratio);
			busy[Kernel_pos]=0;
//			}
		}
		delete kernel_schedule;
		delete need_schedule;
	}
	delete pKernel_list_name;
}
