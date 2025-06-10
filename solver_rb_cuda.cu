#include "indices.cuh"
#include "solver.cuh"
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// Device version of IX macro
#define IX(x, y) (rb_idx((x), (y), (n + 2)))
// Host version of IX macro
#define IX_HOST(x, y) ((x) + (y) * (n + 2))

#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

// CUDA error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t error = call;                                   \
        if (error != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1);                                                \
        }                                                           \
    } while (0)

// CUDA kernel for adding source
__global__ void add_source_kernel(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

// CUDA kernel for setting boundaries
__global__ void set_bnd_kernel(unsigned int n, boundary b, float* x)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= n) {
        x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
}

// CUDA kernel for setting corners
__global__ void set_corners_kernel(unsigned int n, float* x, boundary b)
{
    // Only one thread needed
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Corners
        x[IX_HOST(0, 0)] = 0.5f * (x[IX_HOST(1, 0)] + x[IX_HOST(0, 1)]);
        x[IX_HOST(0, n + 1)] = 0.5f * (x[IX_HOST(1, n + 1)] + x[IX_HOST(0, n)]);
        x[IX_HOST(n + 1, 0)] = 0.5f * (x[IX_HOST(n, 0)] + x[IX_HOST(n + 1, 1)]);
        x[IX_HOST(n + 1, n + 1)] = 0.5f * (x[IX_HOST(n, n + 1)] + x[IX_HOST(n + 1, n)]);
    }
}

// CUDA kernel for red-black solver step
__global__ void lin_solve_rb_step_kernel(grid_color color,
                                         unsigned int n,
                                         float a,
                                         float c,
                                         const float* __restrict__ same0,
                                         const float* __restrict__ neigh,
                                         float* __restrict__ same)
{
    unsigned int width = (n + 2) / 2;
    int shift = color == RED ? 1 : -1;
    unsigned int start = color == RED ? 0 : 1;

    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (y <= n) {
        int local_shift = (y % 2 == 0) ? -shift : shift;
        unsigned int local_start = (y % 2 == 0) ? 1 - start : start;
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + local_start;

        if (x < width - (1 - local_start)) {
            int index = idx(x, y, width);
            same[index] = (same0[index] + a * (neigh[index - width] + neigh[index] + neigh[index + local_shift] + neigh[index + width])) / c;
        }
    }
}

// CUDA kernel for advection
__global__ void advect_kernel(unsigned int n, boundary b, float* d, const float* d0,
                              const float* u, const float* v, float dt)
{
    float dt0 = dt * n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= n && j <= n) {
        float x = i - dt0 * u[IX(i, j)];
        float y = j - dt0 * v[IX(i, j)];

        x = fmaxf(0.5f, fminf(n + 0.5f, x));
        y = fmaxf(0.5f, fminf(n + 0.5f, y));

        int i0 = (int)x;
        int i1 = i0 + 1;
        int j0 = (int)y;
        int j1 = j0 + 1;

        float s1 = x - i0;
        float s0 = 1 - s1;
        float t1 = y - j0;
        float t0 = 1 - t1;

        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

// CUDA kernel for projection
__global__ void project_kernel(unsigned int n, float* u, float* v, float* p, float* div)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= n && j <= n) {
        div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
        p[IX(i, j)] = 0;
    }
}

// CUDA kernel for velocity update
__global__ void update_velocity_kernel(unsigned int n, float* u, float* v, float* p)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= n && j <= n) {
        u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
        v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
}

// Host functions that manage CUDA memory and kernel launches
static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    unsigned int block_size = 256;
    unsigned int num_blocks = (size + block_size - 1) / block_size;

    add_source_kernel<<<num_blocks, block_size>>>(n, x, s, dt);
    CUDA_CHECK(cudaGetLastError());
}

static void set_bnd(unsigned int n, boundary b, float* x)
{
    unsigned int block_size = 256;
    unsigned int num_blocks = (n + block_size - 1) / block_size;
    set_bnd_kernel<<<num_blocks, block_size>>>(n, b, x);
    CUDA_CHECK(cudaGetLastError());
    // Set corners on device
    set_corners_kernel<<<1, 1>>>(n, x, b);
    CUDA_CHECK(cudaGetLastError());
}

static void lin_solve_rb_step(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float* same0,
                              const float* neigh,
                              float* same)
{
    unsigned int width = (n + 2) / 2;
    dim3 block_size(16, 16);
    dim3 num_blocks((width + block_size.x - 1) / block_size.x,
                    (n + block_size.y - 1) / block_size.y);

    lin_solve_rb_step_kernel<<<num_blocks, block_size>>>(color, n, a, c, same0, neigh, same);
    CUDA_CHECK(cudaGetLastError());
}

static void lin_solve(unsigned int n, boundary b,
                      float* x,
                      const float* x0,
                      float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    const float* red0 = x0;
    const float* blk0 = x0 + color_size;
    float* red = x;
    float* blk = x + color_size;

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step(RED, n, a, c, red0, blk, red);
        lin_solve_rb_step(BLACK, n, a, c, blk0, red, blk);
        set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float* x, const float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect(unsigned int n, boundary b, float* d, const float* d0, const float* u, const float* v, float dt)
{
    dim3 block_size(16, 16);
    dim3 num_blocks((n + block_size.x - 1) / block_size.x,
                    (n + block_size.y - 1) / block_size.y);

    advect_kernel<<<num_blocks, block_size>>>(n, b, d, d0, u, v, dt);
    CUDA_CHECK(cudaGetLastError());
    set_bnd(n, b, d);
}

static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
    dim3 block_size(16, 16);
    dim3 num_blocks((n + block_size.x - 1) / block_size.x,
                    (n + block_size.y - 1) / block_size.y);

    project_kernel<<<num_blocks, block_size>>>(n, u, v, p, div);
    CUDA_CHECK(cudaGetLastError());

    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    update_velocity_kernel<<<num_blocks, block_size>>>(n, u, v, p);
    CUDA_CHECK(cudaGetLastError());

    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
}

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
    SWAP(u0, u);
    diffuse(n, VERTICAL, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, HORIZONTAL, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, VERTICAL, u, u0, u0, v0, dt);
    advect(n, HORIZONTAL, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
}