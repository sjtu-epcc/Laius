CC:=gcc
CPP:=g++
CFLAGS:=-Wall -O3 -fopenmp -std=c++0x
LFLAGS:=-ldl -lrt
NV:=nvcc
NVFLAGS:=-arch=sm_70 -O3 --cudart shared

all: bin/init bin/origin bin/cuda_wrap.so bin/libc_wrap.so bin/schedule  bin/uninit #bin/test1
bin/schedule: schedule_new.cpp
	$(CPP)  -I /usr/local/cuda/include $(CFLAGS) $< -o $@ $(LFLAGS) -g
bin/origin: origin.cu
	$(NV) $(NVFLAGS) $< -o $@

bin/cuda_wrap.so: cuda_wrap.cpp *.hpp
	$(CPP) $(CFLAGS) -fPIC -shared $< -o $@ $(LFLAGS)

bin/libc_wrap.so: libc_wrap.cpp *.hpp
	$(CPP) $(CFLAGS) -fPIC -shared $< -o $@ $(LFLAGS)

bin/init: init.cpp *.hpp
	$(CPP) -I /usr/local/cuda/include $(CFLAGS) $< -o $@ $(LFLAGS) -g

bin/test1: bin/test1.cpp
	$(CPP) $(CFLAGS) $< -o $@ $(LFLAGS) -g
bin/uninit: uninit.cpp *.hpp
	$(CPP) $(CFLAGS) $< -o $@ $(LFLAGS) -g
clean:
	-@rm -rf bin/origin bin/cuda_wrap.so bin/libc_wrap.so bin/init bin/schedule bin/uninit
