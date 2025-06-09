#pragma once

#include <stddef.h>
#include <cuda_runtime.h>

typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;
typedef enum { RED,
               BLACK } grid_color;

// CUDA kernel declarations
__global__ void add_source_kernel(unsigned int n, float* x, const float* s, float dt);
__global__ void set_bnd_kernel(unsigned int n, boundary b, float* x);
__global__ void lin_solve_rb_step_kernel(grid_color color,
                                       unsigned int n,
                                       float a,
                                       float c,
                                       const float* __restrict__ same0,
                                       const float* __restrict__ neigh,
                                       float* __restrict__ same);
__global__ void advect_kernel(unsigned int n, boundary b, float* d, const float* d0, 
                             const float* u, const float* v, float dt);
__global__ void project_kernel(unsigned int n, float* u, float* v, float* p, float* div);
__global__ void update_velocity_kernel(unsigned int n, float* u, float* v, float* p);

// Host function declarations
void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt);
void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt);

// Helper function declarations
static void add_source(unsigned int n, float* x, const float* s, float dt);
static void set_bnd(unsigned int n, boundary b, float* x);
static void lin_solve_rb_step(grid_color color,
                            unsigned int n,
                            float a,
                            float c,
                            const float* same0,
                            const float* neigh,
                            float* same);
static void lin_solve(unsigned int n, boundary b,
                     float* x,
                     const float* x0,
                     float a, float c);
static void diffuse(unsigned int n, boundary b, float* x, const float* x0, float diff, float dt);
static void advect(unsigned int n, boundary b, float* d, const float* d0, const float* u, const float* v, float dt);
static void project(unsigned int n, float* u, float* v, float* p, float* div);
