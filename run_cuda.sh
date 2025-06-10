nvcc -arch=sm_86 -c headless.cu -o headless.o

nvcc -arch=sm_86 -c wtime.cpp -o wtime.o

nvcc -arch=sm_86 -c solver_rb_cuda.cu -o solver_rb_cuda.o

nvcc -arch=sm_86 solver_rb_cuda.o wtime.o headless.o -o cuda

./cuda