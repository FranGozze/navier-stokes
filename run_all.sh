#!/bin/bash

# Array of solver files
solvers=(solver_original.c solver_vector1.c solver_vector2.c solver_intrinsics.c)

# Array of compilers
compilers=(gcc)

# Compile flags
FLAGS="-std=c11 -Wall -Wextra -Wno-unused-parameter -march=native -funsafe-math-optimizations -ftree-vectorize -ffast-math -O2"

for compiler in "${compilers[@]}"; do
    echo "Using compiler: $compiler"
    
    $compiler $FLAGS -c headless.c -o headless_${compiler}_-O2.o
    $compiler $FLAGS -c wtime.c -o wtime_${compiler}_-O2.o

    for solver in "${solvers[@]}"; do
        echo "Compiling $solver with $compiler..."
        
        # Get base name without extension
        base=${solver%.c}
        
        # Compile each source file
        $compiler $FLAGS -c "$solver" -o "${base}_${compiler}_-O2.o"
        
        # Link everything together
        $compiler $FLAGS headless_${compiler}_-O2.o "${base}_${compiler}_-O2.o" wtime_${compiler}_-O2.o -o "headless_${compiler}_-O2_${base}"
        
        perf stat -e fp_ret_sse_avx_ops.all ./headless_${compiler}_-O2_${base} 512 > tp2/sv/${compiler}/${base}.csv

        rm headless_${compiler}_-O2_${base}

        echo "Done compiling $solver with $compiler"
        echo "------------------------"
    done
    
    make clean

    echo "Finished with $compiler"
    echo "========================"
done
