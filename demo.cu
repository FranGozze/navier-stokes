/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

    This code is a simple prototype that demonstrates how to use the
    code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
    for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

#include "indices.h"
#include "solver.cuh"
#include "timing.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define IX(x, y) (rb_idx((x), (y), (N + 2)))

static int N;
static float dt, diff, visc;
static float force, source;
static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

// PBO and CUDA interop
static GLuint pbo = 0;
static struct cudaGraphicsResource* cuda_pbo_resource = NULL;
static GLuint tex = 0;

// Host buffers for visualization
static float *h_u = NULL, *h_v = NULL;

static void free_data(void)
{
    if (u) CUDA_CHECK(cudaFree(u));
    if (v) CUDA_CHECK(cudaFree(v));
    if (u_prev) CUDA_CHECK(cudaFree(u_prev));
    if (v_prev) CUDA_CHECK(cudaFree(v_prev));
    if (dens) CUDA_CHECK(cudaFree(dens));
    if (dens_prev) CUDA_CHECK(cudaFree(dens_prev));
    if (h_u) free(h_u);
    if (h_v) free(h_v);
    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
}

static int allocate_data(void)
{
    int size = (N + 2) * (N + 2);
    if (cudaMalloc((void**)&u, size * sizeof(float)) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&v, size * sizeof(float)) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&u_prev, size * sizeof(float)) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&v_prev, size * sizeof(float)) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&dens, size * sizeof(float)) != cudaSuccess) return 0;
    if (cudaMalloc((void**)&dens_prev, size * sizeof(float)) != cudaSuccess) return 0;
    h_u = (float*)malloc(size * sizeof(float));
    h_v = (float*)malloc(size * sizeof(float));
    if (!h_u || !h_v) return 0;

    // Create PBO for density visualization
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    // Create OpenGL texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, N + 2, N + 2, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    return 1;
}

static void clear_data(void)
{
    int size = (N + 2) * (N + 2);
    CUDA_CHECK(cudaMemset(u, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(v, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(u_prev, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(v_prev, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(dens, 0, size * sizeof(float)));
    CUDA_CHECK(cudaMemset(dens_prev, 0, size * sizeof(float)));
    if (h_u) memset(h_u, 0, size * sizeof(float));
    if (h_v) memset(h_v, 0, size * sizeof(float));
}

static void react(float* d, float* u, float* v)
{
    // This is a host-side operation, so copy to host, operate, then copy back
    int size = (N + 2) * (N + 2);
    float *h_u = (float*)malloc(size * sizeof(float));
    float *h_v = (float*)malloc(size * sizeof(float));
    float *h_d = (float*)malloc(size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_u, u, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, v, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_d, d, size * sizeof(float), cudaMemcpyDeviceToHost));
    float max_velocity2 = 0.0f, max_density = 0.0f;
    for (int i = 0; i < size; i++) {
        float vel2 = h_u[i] * h_u[i] + h_v[i] * h_v[i];
        if (max_velocity2 < vel2) max_velocity2 = vel2;
        if (max_density < h_d[i]) max_density = h_d[i];
    }
    for (int i = 0; i < size; i++) {
        h_u[i] = h_v[i] = h_d[i] = 0.0f;
    }
    if (max_velocity2 < 0.0000005f) {
        h_u[IX(N / 2, N / 2)] = force * 10.0f;
        h_v[IX(N / 2, N / 2)] = force * 10.0f;
    }
    if (max_density < 1.0f) {
        h_d[IX(N / 2, N / 2)] = source * 10.0f;
    }
    CUDA_CHECK(cudaMemcpy(u, h_u, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v, h_v, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d, h_d, size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_u); free(h_v); free(h_d);
}

static void one_step(void)
{
    static int times = 1;
    static double start_t = 0.0;
    static double one_second = 0.0;
    static double react_ns_p_cell = 0.0;
    static double vel_ns_p_cell = 0.0;
    static double dens_ns_p_cell = 0.0;

    start_t = wtime();
    react(dens_prev, u_prev, v_prev);
    react_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    vel_step(N, u, v, u_prev, v_prev, visc, dt);
    vel_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

    start_t = wtime();
    dens_step(N, dens, dens_prev, u, v, diff, dt);
    dens_ns_p_cell += 1.0e9 * (wtime() - start_t) / (N * N);

    if (1.0 < wtime() - one_second) {
        printf("%lf, %lf, %lf, %lf\n",
               (react_ns_p_cell + vel_ns_p_cell + dens_ns_p_cell) / times,
               react_ns_p_cell / times, vel_ns_p_cell / times, dens_ns_p_cell / times);
        one_second = wtime();
        react_ns_p_cell = 0.0;
        vel_ns_p_cell = 0.0;
        dens_ns_p_cell = 0.0;
        times = 1;
    } else {
        times++;
    }
}

static void draw_velocity(void)
{
    int i, j;
    float x, y, h;
    int size = (N + 2) * (N + 2);
    CUDA_CHECK(cudaMemcpy(h_u, u, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v, v, size * sizeof(float), cudaMemcpyDeviceToHost));
    h = 1.0f / N;
    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (i = 1; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 1; j <= N; j++) {
            y = (j - 0.5f) * h;
            glVertex2f(x, y);
            glVertex2f(x + h_u[IX(i, j)], y + h_v[IX(i, j)]);
        }
    }
    glEnd();
}

static void draw_density(void)
{
    int size = (N + 2) * (N + 2);
    // Map PBO for CUDA access
    float* d_pbo = NULL;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &num_bytes, cuda_pbo_resource);
    // Copy density from simulation device memory to PBO device memory
    CUDA_CHECK(cudaMemcpy(d_pbo, dens, size * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    // Draw PBO as texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N + 2, N + 2, GL_LUMINANCE, GL_FLOAT, 0);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

int main(int argc, char** argv)
{
    if (argc != 1 && argc != 7) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
        fprintf(stderr, "where:\n");
        fprintf(stderr, "\t N      : grid resolution\n");
        fprintf(stderr, "\t dt     : time step\n");
        fprintf(stderr, "\t diff   : diffusion rate of the density\n");
        fprintf(stderr, "\t visc   : viscosity of the fluid\n");
        fprintf(stderr, "\t force  : scales the mouse movement that generate a force\n");
        fprintf(stderr, "\t source : amount of density that will be deposited\n");
        exit(1);
    }
    if (argc == 1) {
        N = 64;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
                N, dt, diff, visc, force, source);
    } else {
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        diff = atof(argv[3]);
        visc = atof(argv[4]);
        force = atof(argv[5]);
        source = atof(argv[6]);
    }
    if (!allocate_data()) exit(1);
    clear_data();
    for (int i = 0; i < 2048; i++) {
        one_step();
    }
    free_data();
    exit(0);
}
