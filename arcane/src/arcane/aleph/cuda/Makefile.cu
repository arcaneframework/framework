CUDA_eros=/opt/cuda/3.2
ARCH_eros=sm_21

CUDA_lascaux=/opt/cuda/5.0
ARCH_lascaux=sm_23

CUDA_x-airen=/usr/local/opendev1/gcc/cuda/2.3
ARCH_x-airen=sm_13

CUDA_HOME = $(CUDA_$(shell cea_machine))

NVCC_FLAGS = -shared -arch=$(ARCH_$(shell cea_machine)) --compiler-options -fPIC -DARCANE_NOEXCEPT_FOR_NVCC
NVCC = $(CUDA_HOME)/bin/nvcc $(NVCC_FLAGS)


CCFLAGS = -Wall -O2  -malign-double -fPIC
CC = gcc $(CCFLAGS)


all: libcnccuda 

libcnccuda: AlephCuda.o
	$(NVCC) -lib -o libAlephCuda.so AlephCuda.o


#############
%.o : %.cu
	@echo "NVIDIA-Compiling '$@'"
	$(NVCC) -c $< $(CFLAGS) -o $@


#############
%.o: %.cpp
	@echo "HOST-Compiling '$@'"
	$(CC) -c -I$(CUDA_HOME)/include -o $@ $*.cpp

cln:
	@-rm -f *.o *.a *.so *.co
