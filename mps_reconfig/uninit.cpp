#include "ipc.hpp"

int main()
{
	vector<string> processName = getProcessName();
	int ret;
	for(int i=0;i<processName.size();i++)
	{
		ret = shm_unlink(processName[i].c_str());
		assert(ret!=-1);
	}
	ret = shm_unlink("shared_compute");
}
