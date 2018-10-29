#include "main.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "NBody.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>


double getRandom(double min, double max)
{
	double r = (double)rand() / RAND_MAX;
	return r*(max - min) + min;
}

__global__ void addTrail(unsigned char* d_buffer,
        const double decay,
        const unsigned int screen_width,
        const unsigned int screen_height)

{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= screen_width || y >= screen_height) return;
    d_buffer[x * screen_height * 3 + y * 3 + 0] *= decay;
    d_buffer[x * screen_height * 3 + y * 3 + 1] *= decay;
    d_buffer[x * screen_height * 3 + y * 3 + 2] *= decay;
}

__global__ void updateFrame(struct body *d_bodies,
        unsigned char* d_buffer,
        unsigned int screen_width,
        unsigned int screen_height,
        unsigned int num_bodies)
{
    unsigned int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= num_bodies) return;

    unsigned int x = (unsigned int)((d_bodies[i].x + 1.0)/2.0 * screen_width);
    unsigned int y = (unsigned int)((d_bodies[i].y + 1.0)/2.0 * screen_height);
    d_buffer[x * screen_height * 3 + y * 3 + 0] = 255;
    d_buffer[x * screen_height * 3 + y * 3 + 1] = 255;
    d_buffer[x * screen_height * 3 + y * 3 + 2] = 255;
}

void rasterize(struct body* d_bodies, unsigned char* d_buffer, unsigned char* h_buffer)
{
	/**
	rasterize the bodies from x,y: (-1,-1) to (1,1) according to some kind of formula

	Note: You can change the code for better visualization
	As the following code can be parallelized, you can optimize this routine with CUDA.

	\param bodies A collection of bodies (located on the device).
	\param buffer the RGB buffer for screen display (located on the host).
	*/
    unsigned int bufferSize = SCREEN_WIDTH * SCREEN_HEIGHT * 3 * sizeof(unsigned char);

    // add trail effect
    dim3 gridShape(1, SCREEN_HEIGHT);
    dim3 blockShape((SCREEN_WIDTH + gridShape.x - 1) / gridShape.x,
                    (SCREEN_HEIGHT + gridShape.y - 1) / gridShape.y);
    addTrail<<<gridShape, blockShape>>>(d_buffer, 0.98, SCREEN_WIDTH, SCREEN_HEIGHT);

    // update frame
    unsigned int groupSize = 8;
    updateFrame<<<groupSize, (NUM_BODIES + groupSize - 1) / groupSize>>>(
            d_bodies, d_buffer, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_BODIES);

    cudaMemcpy(h_buffer, d_buffer, bufferSize, cudaMemcpyDeviceToHost);
}

void initializeNBodyCuda(struct body * &d_bodies, unsigned char* &d_buffer)
{
	/**
	initialize the bodies, then copy to the CUDA device memory
	return the device pointer so that it can be reused in the NBodyTimestepCuda function.
	*/
	// initialize the position and velocity
	// you can implement own initial conditions to form a sprial/ellipse galaxy, have fun.
	struct body* h_bodies = new struct body[NUM_BODIES];
	for(int i = 0; i < NUM_BODIES; i++)
    {
		h_bodies[i].x = getRandom(-1.0, 1.0);
        h_bodies[i].y = getRandom(-1.0, 1.0);
        h_bodies[i].vx = getRandom(-1.0, 1.0);
        h_bodies[i].vy = getRandom(-1.0, 1.0);
        //bodies[i].vx = 0;
        //bodies[i].vy = 0;
        h_bodies[i].m = getRandom(1e5, 1e7);
    }
    //d_body points to device memory
	cudaMalloc((void**)&d_bodies, NUM_BODIES * sizeof(struct body));
	cudaMemcpy(d_bodies, h_bodies, NUM_BODIES * sizeof(struct body), cudaMemcpyHostToDevice);
	// after copy initialized bodies to device, we can recycle host memory now
	delete [] h_bodies;

	// initialize buffer
    unsigned int bufferSize = SCREEN_WIDTH * SCREEN_HEIGHT * 3 * sizeof(unsigned char);
    cudaMalloc((void**)&d_buffer, bufferSize);
    cudaMemset(d_buffer, 0, bufferSize);
}

void freeMem(struct body* d_bodies, unsigned char* d_buffer)
{
    // if bodies is located in host memory, use delete
    // if in device memory, use cudaFree
    cudaFree(d_buffer);
    cudaFree(d_bodies);
}


__global__ void updateBodies(struct body* d_bodies,
        double dt,
        double rx,
        double ry,
        bool cursor)
{
    __shared__ double ax[NUM_BODIES], ay[NUM_BODIES];
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    // initialization
    ax[tid] = 0;
    ay[tid] = 0;
    __syncthreads();


    if(tid != bid) 
	{
        double dx = d_bodies[tid].x - d_bodies[bid].x;
        double dy = d_bodies[tid].y - d_bodies[bid].y;
        double dist3 = pow(dx * dx + dy * dy, 1.5) + eps;
        ax[tid] = G * d_bodies[tid].m / dist3 * dx;
        ay[tid] = G * d_bodies[tid].m / dist3 * dy;
	}
	__syncthreads();
    /* TODO: here we assume NUM_BODIES is 2^k for some k,
      or else we need extra judgement for reduction */
	for (unsigned int s = NUM_BODIES >> 1; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			ax[tid] += ax[tid + s];
			ay[tid] += ay[tid + s];
		}
		__syncthreads();
	}
    
    // TODO: try seperate the following code to another kernel
    // because in remaining part, there is only one thread alive on blocks
    if(tid == 0)
    {
        if(cursor)
        {
            double dx = rx - d_bodies[bid].x;
            double dy = ry - d_bodies[bid].y;
            double dist3 = pow(dx * dx + dy * dy, 1.5) + eps;
            // magnify the effect of cursor
            ax[0] += 1e7 * G * cursor_weight / dist3 * dx;
            ay[0] += 1e7 * G * cursor_weight / dist3 * dy;
        }
        double vx = d_bodies[bid].vx + ax[0] * dt;
        double vy = d_bodies[bid].vy + ay[0] * dt;
        double x = d_bodies[bid].x + vx * dt;
        double y = d_bodies[bid].y + vy * dt;
        // after collsision, the actual positon should
        // be symmetric correspoding to boundary
        // here we assume the velocity won't be too large,
        // so at most one collision can happen between two frames
        if(x > 1.0)
        {
           vx *= -collision_damping;
           // more precicsely, x = 1.6 - 0.6*x;
           x = 2.0 - x;
        }
        else if(x < -1.0)
        {
            vx *= -collision_damping;
            x = -2.0 -x;
        }
        if(y > 1.0)
        {
            vy *= -collision_damping;
            y = 2.0 - y;
        }
        else if(y < -1.0)
        {
            vy *= -collision_damping;
            y = -2.0 -y;
        }
        d_bodies[bid].x = x;
        d_bodies[bid].y = y;
        d_bodies[bid].vx = vx;
        d_bodies[bid].vy = vy;
    }
}

void NBodyTimestepCuda(struct body* d_bodies, double rx, double ry, bool cursor)
{
    /**
    Compute a time step on the CUDA device.
    \param bodies A collection of bodies (located on the device).
    \param rx position x of the cursor.
    \param ry position y of the cursor.
    \param cursor Enable the mouse interaction if true (adding a weight = cursor_weight body in the computation).
    */
    // TODO: current version has low scalability, try to write new kernel to hadle more bodies
    assert(NUM_BODIES <= 1024);
    double dt = 1e-3;
    updateBodies<<<NUM_BODIES, NUM_BODIES>>>(d_bodies, dt, rx, ry, cursor);
}
