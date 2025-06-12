
ARCH=sm_86

nvcc -arch=$ARCH -c demo.cu -o demo.o

nvcc -arch=$ARCH -c wtime.cpp -o wtime.o

nvcc -arch=$ARCH -c solver_rb_pragma.cu -o solver_rb_cuda.o

nvcc -arch=$ARCH solver_rb_cuda.o wtime.o demo.o -o demo -lGL -lGLU -lglut

./demo

rm demo.o wtime.o solver_rb_cuda.o demo