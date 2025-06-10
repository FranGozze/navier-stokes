nvcc -arch=sm_86 -c headless.cu -o headless.o

nvcc -arch=sm_86 -c wtime.cpp -o wtime.o

nvcc -arch=sm_86 solver_rb_cuda.cu_nvcc_-O2.o wtime.o headless.o -o cuda

./cuda