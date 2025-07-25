#include <cuda_runtime.h>
#include <omp.h>
#include <stddef.h>

#include "indices.cuh"
#include "solver.h"

#define IX(x, y) (rb_idx((x), (y), (n + 2)))
#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;
typedef enum { RED,
               BLACK } grid_color;

__global__ void add_source_kernel(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}


static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    unsigned int block_size = 256;
    unsigned int num_blocks = (size + block_size - 1) / block_size;

    add_source_kernel<<<num_blocks, block_size>>>(n, x, s, dt);
    cudaDeviceSynchronize();
}

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

__global__ void set_bnd_corners_kernel(unsigned int n, float* x)
{
    if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0) {
        x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
        x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
        x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
        x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
    }
}


static void set_bnd(unsigned int n, boundary b, float* x)
{
    unsigned int block_size = 256;
    unsigned int num_blocks = (n + block_size - 1) / block_size;

    set_bnd_kernel<<<num_blocks, block_size>>>(n, b, x);
    cudaDeviceSynchronize();
    // Handle corners in a single thread
    dim3 block_size_corners(1, 1);
    dim3 num_blocks_corners(1, 1);
    set_bnd_corners_kernel<<<num_blocks_corners, block_size_corners>>>(n, x);
    cudaDeviceSynchronize();
    // Ensure corners are set correctly
    // x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    // x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    // x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
    // x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

// CUDA kernel for red-black solver step
#define BLOCK_SIZE_X 16 // Warp-aligned
#define BLOCK_SIZE_Y 16 // Puedes ajustar esto según ocupación

__global__ void lin_solve_rb_step_shared_kernel(
    grid_color color,
    unsigned int n,
    float a,
    float c,
    const float* __restrict__ same0,
    const float* __restrict__ neigh,
    float* __restrict__ same)
{
    extern __shared__ float shared_neigh[];

    unsigned int width = (n + 2) / 2;
    int shift = color == RED ? 1 : -1;

    // Coordenadas del bloque
    unsigned int block_start_x = blockIdx.x * (blockDim.x - 2);
    unsigned int block_start_y = blockIdx.y * (blockDim.y - 2) + 1;

    // Coordenadas locales con halo
    unsigned int local_x = threadIdx.x;
    unsigned int local_y = threadIdx.y;

    // Coordenadas globales con halo
    unsigned int global_x = block_start_x + local_x - 1;
    unsigned int global_y = block_start_y + local_y - 1;

    // Cargar datos a shared memory (incluyendo halo)
    if (global_y <= n && global_x < width) {
        int load_idx = idx(global_x, global_y, width);
        shared_neigh[local_y * blockDim.x + local_x] = neigh[load_idx];
    }

    __syncthreads();

    // Solo los hilos internos (no halo) calculan resultados
    if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 && threadIdx.y > 0 && threadIdx.y < blockDim.y - 1) {

        unsigned int x = block_start_x + threadIdx.x - 1;
        unsigned int y = block_start_y + threadIdx.y - 1;

        // Aplicar patrón red-black
        int start = color == RED ? 0 : 1;
        int pattern_shift = (y % 2) ? shift : -shift;
        int pattern_start = (y % 2) ? start : 1 - start;

        // Verificar que cumple con el patrón
        if ((x - pattern_start) % 2 != 0 || y > n || x >= width - (1 - pattern_start))
            return;

        int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int s_top = local_idx - blockDim.x;
        int s_bottom = local_idx + blockDim.x;
        int s_side = local_idx + pattern_shift;

        int global_idx = idx(x, y, width);

        same[global_idx] = (same0[global_idx] + a * (shared_neigh[s_top] + shared_neigh[local_idx] + shared_neigh[s_side] + shared_neigh[s_bottom])) / c;
    }
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

    // int shift = color == RED ? 1 : -1;
    // unsigned int start = color == RED ? 0 : 1;

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (n + blockSize.y - 1) / blockSize.y);

    // Calcular tamaño de shared memory (halo de 1 elemento alrededor del bloque)
    size_t sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(float);

    lin_solve_rb_step_shared_kernel<<<gridSize, blockSize, sharedMemSize>>>(color, n, a, c, same0, neigh, same);
    cudaDeviceSynchronize();


    // for (unsigned int y = 1; y <= n; ++y) {
    //     int local_shift = (y % 2 == 0) ? -shift : shift;
    //     unsigned int local_start = (y % 2 == 0) ? 1 - start : start;
    //     for (unsigned int x = local_start; x < width - (1 - local_start); ++x) {
    //         int index = idx(x, y, width);
    //         same[index] = (same0[index] + a * (neigh[index - width] + neigh[index] + neigh[index + local_shift] + neigh[index + width])) / c;
    //     }
    // }
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


static void advect(unsigned int n, boundary b, float* d, const float* d0, const float* u, const float* v, float dt)
{
    dim3 block_size(16, 16);
    dim3 num_blocks((n + block_size.x - 1) / block_size.x,
                    (n + block_size.y - 1) / block_size.y);

    advect_kernel<<<num_blocks, block_size>>>(n, b, d, d0, u, v, dt);
    cudaDeviceSynchronize();
    set_bnd(n, b, d);
}

__global__ void project_kernel(unsigned int n, float* u, float* v, float* p, float* div)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= n && j <= n) {
        div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
        p[IX(i, j)] = 0;
    }
}

__global__ void update_velocity_kernel(unsigned int n, float* u, float* v, float* p)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= n && j <= n) {
        u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
        v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
}


static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
    dim3 block_size(16, 16);
    dim3 num_blocks((n + block_size.x - 1) / block_size.x,
                    (n + block_size.y - 1) / block_size.y);

    project_kernel<<<num_blocks, block_size>>>(n, u, v, p, div);
    cudaDeviceSynchronize();
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    update_velocity_kernel<<<num_blocks, block_size>>>(n, u, v, p);
    cudaDeviceSynchronize();
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
