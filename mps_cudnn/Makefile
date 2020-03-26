CC:=gcc
CPP:=g++
CFLAGS:=-Wall -O3 -fopenmp -std=c++0x
LFLAGS:=-ldl -lrt
NV:=nvcc
NVFLAGS:=-arch=sm_50 -O3 --cudart shared

all: bin/init bin/origin bin/cuda_wrap.so bin/libc_wrap.so

bin/origin: origin.cu
	$(NV) $(NVFLAGS) $< -o $@

bin/cuda_wrap.so: cuda_wrap.cpp *.h
	$(CPP) $(CFLAGS) -fPIC -shared $< -o $@ $(LFLAGS)

bin/libc_wrap.so: libc_wrap.cpp *.h
	$(CPP) $(CFLAGS) -fPIC -shared $< -o $@ $(LFLAGS)

bin/init: init.cpp *.h
	$(CPP) $(CFLAGS) $< -o $@ $(LFLAGS)

clean:
	-@rm -rf bin/origin bin/cuda_wrap.so bin/libc_wrap.so bin/init
