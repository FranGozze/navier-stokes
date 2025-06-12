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

#include <GL/glut.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "indices.h"
#include "solver.h"
#include "timing.h"

/* macros */

#define IX(x, y) (rb_idx((x), (y), (N + 2)))

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data(void)
{
    if (u)
        cudaFree(u);
    if (v)
        cudaFree(v);
    if (u_prev)
        cudaFree(u_prev);
    if (v_prev)
        cudaFree(v_prev);
    if (dens)
        cudaFree(dens);
    if (dens_prev)
        cudaFree(dens_prev);
}

static void clear_data(void)
{
    int size = (N + 2) * (N + 2);
    cudaMemset(u, 0, size * sizeof(float));
    cudaMemset(v, 0, size * sizeof(float));
    cudaMemset(u_prev, 0, size * sizeof(float));
    cudaMemset(v_prev, 0, size * sizeof(float));
    cudaMemset(dens, 0, size * sizeof(float));
    cudaMemset(dens_prev, 0, size * sizeof(float));
}

static int allocate_data(void)
{
    int size = (N + 2) * (N + 2);

    cudaMalloc((void**)&u, size * sizeof(float));
    cudaMalloc((void**)&v, size * sizeof(float));
    cudaMalloc((void**)&u_prev, size * sizeof(float));
    cudaMalloc((void**)&v_prev, size * sizeof(float));
    cudaMalloc((void**)&dens, size * sizeof(float));
    cudaMalloc((void**)&dens_prev, size * sizeof(float));

    if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev) {
        fprintf(stderr, "cannot allocate data\n");
        return (0);
    }

    return (1);
}


/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display(void)
{
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

static void post_display(void)
{
    glutSwapBuffers();
}

static void draw_velocity(void)
{
    int i, j;
    float x, y, h;

    h = 1.0f / N;

    // Copy u and v from device to host
    int size = (N + 2) * (N + 2);
    static float* u_host = NULL;
    static float* v_host = NULL;
    if (!u_host)
        u_host = (float*)malloc(size * sizeof(float));
    if (!v_host)
        v_host = (float*)malloc(size * sizeof(float));
    cudaMemcpy(u_host, u, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_host, v, size * sizeof(float), cudaMemcpyDeviceToHost);

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(1.0f);

    glBegin(GL_LINES);

    for (i = 1; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 1; j <= N; j++) {
            y = (j - 0.5f) * h;

            glVertex2f(x, y);
            glVertex2f(x + u_host[IX(i, j)], y + v_host[IX(i, j)]);
        }
    }

    glEnd();
}

static void draw_density(void)
{
    int i, j;
    float x, y, h, d00, d01, d10, d11;

    h = 1.0f / N;

    // Copy dens from device to host
    int size = (N + 2) * (N + 2);
    static float* dens_host = NULL;
    if (!dens_host) {
        dens_host = (float*)malloc(size * sizeof(float));
    }
    cudaMemcpy(dens_host, dens, size * sizeof(float), cudaMemcpyDeviceToHost);

    glBegin(GL_QUADS);

    for (i = 0; i <= N; i++) {
        x = (i - 0.5f) * h;
        for (j = 0; j <= N; j++) {
            y = (j - 0.5f) * h;

            d00 = dens_host[IX(i, j)];
            d01 = dens_host[IX(i, j + 1)];
            d10 = dens_host[IX(i + 1, j)];
            d11 = dens_host[IX(i + 1, j + 1)];

            glColor3f(d00, d00, d00);
            glVertex2f(x, y);
            glColor3f(d10, d10, d10);
            glVertex2f(x + h, y);
            glColor3f(d11, d11, d11);
            glVertex2f(x + h, y + h);
            glColor3f(d01, d01, d01);
            glVertex2f(x, y + h);
        }
    }

    glEnd();
}

/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/


// Device kernel to compute max velocity^2 and max density
__global__ void compute_max_kernel(const float* d, const float* u, const float* v, int size, float* max_velocity2, float* max_density)
{
    extern __shared__ float sdata[];
    float* smax_vel2 = sdata;
    float* smax_dens = sdata + blockDim.x;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_vel2 = 0.0f;
    float local_dens = 0.0f;

    if (i < size) {
        local_vel2 = u[i] * u[i] + v[i] * v[i];
        local_dens = d[i];
    }

    smax_vel2[tid] = local_vel2;
    smax_dens[tid] = local_dens;
    __syncthreads();

    // Parallel reduction for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smax_vel2[tid] < smax_vel2[tid + s])
                smax_vel2[tid] = smax_vel2[tid + s];
            if (smax_dens[tid] < smax_dens[tid + s])
                smax_dens[tid] = smax_dens[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_velocity2[blockIdx.x] = smax_vel2[0];
        max_density[blockIdx.x] = smax_dens[0];
    }
}

// Device kernel to clear arrays
__global__ void clear_arrays_kernel(float* d, float* u, float* v, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d[i] = 0.0f;
        u[i] = 0.0f;
        v[i] = 0.0f;
    }
}

// Device kernel to set a value at a specific index
__global__ void set_value_kernel(float* arr, int idx, float value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        arr[idx] = value;
    }
}

static void react(float* d, float* u, float* v)
{
    int size = (N + 2) * (N + 2);

    // Compute max velocity^2 and max density on device
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    float* d_max_velocity2;
    float* d_max_density;
    cudaMalloc(&d_max_velocity2, blocks * sizeof(float));
    cudaMalloc(&d_max_density, blocks * sizeof(float));

    compute_max_kernel<<<blocks, threads, threads * 2 * sizeof(float)>>>(d, u, v, size, d_max_velocity2, d_max_density);

    // Reduce on host
    float* h_max_velocity2 = (float*)malloc(blocks * sizeof(float));
    float* h_max_density = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_max_velocity2, d_max_velocity2, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_density, d_max_density, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float max_velocity2 = 0.0f;
    float max_density = 0.0f;
    for (int i = 0; i < blocks; i++) {
        if (max_velocity2 < h_max_velocity2[i])
            max_velocity2 = h_max_velocity2[i];
        if (max_density < h_max_density[i])
            max_density = h_max_density[i];
    }

    free(h_max_velocity2);
    free(h_max_density);
    cudaFree(d_max_velocity2);
    cudaFree(d_max_density);

    // Clear arrays on device
    clear_arrays_kernel<<<blocks, threads>>>(d, u, v, size);

    // Set initial values if needed
    if (max_velocity2 < 0.0000005f) {
        int idx = IX(N / 2, N / 2);
        set_value_kernel<<<1, 1>>>(u, idx, force * 10.0f);
        set_value_kernel<<<1, 1>>>(v, idx, force * 10.0f);
    }
    if (max_density < 1.0f) {
        int idx = IX(N / 2, N / 2);
        set_value_kernel<<<1, 1>>>(d, idx, source * 10.0f);
    }

    // Handle mouse input on host, then update device arrays
    if (!mouse_down[0] && !mouse_down[2])
        return;

    int i = (int)((mx / (float)win_x) * N + 1);
    int j = (int)(((win_y - my) / (float)win_y) * N + 1);

    if (i < 1 || i > N || j < 1 || j > N)
        return;

    if (mouse_down[0]) {
        float u_val = force * (mx - omx);
        float v_val = force * (omy - my);
        int idx = IX(i, j);
        set_value_kernel<<<1, 1>>>(u, idx, u_val);
        set_value_kernel<<<1, 1>>>(v, idx, v_val);
    }

    if (mouse_down[2]) {
        int idx = IX(i, j);
        set_value_kernel<<<1, 1>>>(d, idx, source);
    }

    omx = mx;
    omy = my;
}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func(unsigned char key, int x, int y)
{
    switch (key) {
    case 'c':
    case 'C':
        clear_data();
        break;

    case 'q':
    case 'Q':
        free_data();
        exit(0);
        break;

    case 'v':
    case 'V':
        dvel = !dvel;
        break;
    }
}

static void mouse_func(int button, int state, int x, int y)
{
    omx = mx = x;
    omx = my = y;

    mouse_down[button] = state == GLUT_DOWN;
}

static void motion_func(int x, int y)
{
    mx = x;
    my = y;
}

static void reshape_func(int width, int height)
{
    glutSetWindow(win_id);
    glutReshapeWindow(width, height);

    win_x = width;
    win_y = height;
}

static void idle_func(void)
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

    if (1.0 < wtime() - one_second) { /* at least 1s between stats */
        printf("%lf, %lf, %lf, %lf: ns per cell total, react, vel_step, dens_step\n",
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

    glutSetWindow(win_id);
    glutPostRedisplay();
}

static void display_func(void)
{
    pre_display();

    if (dvel)
        draw_velocity();
    else
        draw_density();

    post_display();
}


/*
  ----------------------------------------------------------------------
   open_glut_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glut_window(void)
{
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    win_id = glutCreateWindow("Alias | wavefront");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();

    pre_display();

    glutKeyboardFunc(key_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char** argv)
{
    glutInit(&argc, argv);

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

    printf("\n\nHow to use this demo:\n\n");
    printf("\t Add densities with the right mouse button\n");
    printf("\t Add velocities with the left mouse button and dragging the mouse\n");
    printf("\t Toggle density/velocity display with the 'v' key\n");
    printf("\t Clear the simulation by pressing the 'c' key\n");
    printf("\t Quit by pressing the 'q' key\n");

    dvel = 0;

    if (!allocate_data())
        exit(1);
    clear_data();

    win_x = 512;
    win_y = 512;
    open_glut_window();

    glutMainLoop();

    exit(0);
}
