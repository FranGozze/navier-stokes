
ARCH=sm_86

nvcc -arch=$ARCH -c headless.cu -o headless.o

nvcc -arch=$ARCH -c wtime.cpp -o wtime.o

nvcc -arch=$ARCH -c solver_rb_pragma.cu -o solver_rb_cuda.o

nvcc -arch=$ARCH solver_rb_cuda.o wtime.o headless.o -o cuda

./cuda

rm headless.o wtime.o solver_rb_cuda.o cuda