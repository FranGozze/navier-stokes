#include <stddef.h>

#include "solver.h"

#define GROUP_SIZE 8
#define IX(i, j) ((i) + (n + 2) * (j))
#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;

static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float* x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }

    // posible vectorization
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}


static void lin_solve(unsigned int n, boundary b, float* x, const float* x0, float a, float c)
{
    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int j = 1; j <= n; j++) {
            unsigned int i = 1;
            for (; i <= n - (GROUP_SIZE - 1); i += GROUP_SIZE) {
                float x_up[GROUP_SIZE] = {x[IX(i, j-1)], x[IX(i+1, j-1)], x[IX(i+2, j-1)], x[IX(i+3, j-1)],x[IX(i+4, j-1)], x[IX(i+5, j-1)], x[IX(i+6, j-1)], x[IX(i+7, j-1)]};
                float x_left[GROUP_SIZE] = {x[IX(i-1, j)], x[IX(i, j)], x[IX(i+1, j)], x[IX(i+2, j)],x[IX(i+3, j)], x[IX(i+4, j)], x[IX(i+5, j)], x[IX(i+6, j)]};
                float x_right[GROUP_SIZE] = {x[IX(i+1, j)], x[IX(i+2, j)], x[IX(i+3, j)], x[IX(i+4, j)],x[IX(i+5, j)], x[IX(i+6, j)], x[IX(i+7, j)], x[IX(i+8, j)]};
                float x_down[GROUP_SIZE] = {x[IX(i, j+1)], x[IX(i+1, j+1)], x[IX(i+2, j+1)], x[IX(i+3, j+1)],x[IX(i+4, j+1)], x[IX(i+5, j+1)], x[IX(i+6, j+1)], x[IX(i+7, j+1)]};
                
                float x0_vals[GROUP_SIZE] = {x0[IX(i, j)], x0[IX(i+1, j)], x0[IX(i+2, j)], x0[IX(i+3, j)],x0[IX(i+4, j)], x0[IX(i+5, j)], x0[IX(i+6, j)], x0[IX(i+7, j)]};
                
                float temp[GROUP_SIZE];
                for (int m = 0; m < GROUP_SIZE; m++) {
                    temp[m] = (x0_vals[m] + a * (x_left[m] + x_right[m] + x_up[m] + x_down[m])) / c;
                }
                for (int m = 0; m < GROUP_SIZE; m++) {
                    x[IX(i+m, j)] = temp[m];
                }
            }
            
            for (; i <= n; i++) {
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i-1, j)] + x[IX(i+1, j)] + 
                                                  x[IX(i, j-1)] + x[IX(i, j+1)])) / c;
            }
        }
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
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;
    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 1; i <= n; i++) {
            x = i - dt0 * u[IX(i, j)];
            y = j - dt0 * v[IX(i, j)];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            i0 = (int)x;
            i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            j0 = (int)y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(n, b, d);
}

static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
    float constant = -0.5f / n;
    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 1; i <= n; i++) {
            // possible vectorization
            div[IX(i, j)] = constant * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);
    float constant2 = 0.5f * n;
    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 1; i <= n; i++) {
            // possible vectorization
            u[IX(i, j)] -= constant2 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= constant2 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
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
