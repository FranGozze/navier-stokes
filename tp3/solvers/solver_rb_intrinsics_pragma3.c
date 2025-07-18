#include <immintrin.h> // For AVX intrinsics
#include <stddef.h>


#include "indices.h"
#include "math.h"
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

static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
#pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float* x)
{
#pragma omp parallel for schedule(static)
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
#pragma omp single
    {
        x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
        x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
        x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
        x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
    }
}


static void lin_solve_rb_step(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float* restrict same0,
                              const float* restrict neigh,
                              float* restrict same)
{
    int shift = color == RED ? 1 : -1;
    unsigned int start = color == RED ? 0 : 1;

    unsigned int width = (n + 2) / 2;
    __m256 a_vec = _mm256_set1_ps(a);
    __m256 c_vec = _mm256_set1_ps(c);

#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int y = 1; y <= n; ++y) {
        int local_shift = (y % 2 == 0) ? -shift : shift;
        unsigned int local_start = (y % 2 == 0) ? 1 - start : start;

        for (unsigned int x = local_start; x < width - (1 - local_start); x += 8) {
            // x + y * width
            int index = idx(x, y, width);
            __m256 x_up = _mm256_loadu_ps(&neigh[index - width]);
            __m256 x_left = _mm256_loadu_ps(&neigh[index]);
            __m256 x_right = _mm256_loadu_ps(&neigh[index + local_shift]);
            __m256 x_down = _mm256_loadu_ps(&neigh[index + width]);
            __m256 sum = _mm256_add_ps(x_left, x_right);
            sum = _mm256_add_ps(sum, x_up);
            sum = _mm256_add_ps(sum, x_down);
            __m256 same_vals = _mm256_loadu_ps(&same0[index]);
            __m256 result = _mm256_add_ps(same_vals, _mm256_mul_ps(a_vec, sum));
            result = _mm256_div_ps(result, c_vec);
            _mm256_storeu_ps(&same[index], result);
        }
    }
}
static void lin_solve(unsigned int n, boundary b,
                      float* restrict x,
                      const float* restrict x0,
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
    float dt0 = dt * n;
#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            float x = i - dt0 * u[IX(i, j)];
            float y = j - dt0 * v[IX(i, j)];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            int i0 = (int)x;
            int i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            int j0 = (int)y;
            int j1 = j0 + 1;
            float s1 = x - i0;
            float s0 = 1 - s1;
            float t1 = y - j0;
            float t0 = 1 - t1;
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(n, b, d);
}

static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

#pragma omp parallel for collapse(2) schedule(static)
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
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
