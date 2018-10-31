#include "main.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "NBody.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>


float getRandom(float min, float max)
{
	float r = (float)rand() / RAND_MAX;
	return r*(max - min) + min;
}

__global__ void addTrail(unsigned char* d_buffer,
        const float decay,
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
    addTrail<<<gridShape, blockShape>>>(d_buffer, 0, SCREEN_WIDTH, SCREEN_HEIGHT);

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


// TODO: use the anti-symmetry property of force matrix to reduce computation
__global__ void scalableUpdate(struct body* d_bodies,
        float dt,
        float rx,
        float ry,
        bool cursor,
        unsigned int num_bodies)
{
    extern __shared__ body sbodies[];
    float ax=0, ay=0;
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if(offset >= num_bodies) return;

    body myBody = d_bodies[offset];

    // this part can be accelerated
    for(unsigned int i = 0; i < gridDim.x; i++)
    {
        sbodies[threadIdx.x] = d_bodies[i * blockDim.x + threadIdx.x];
        __syncthreads();
        for(unsigned int j = 0; j < blockDim.x; j++)
        {
            body otherBody = sbodies[j];
            float dx = otherBody.x - myBody.x;
            float dy = otherBody.y - myBody.y;
            float dist3 = pow(dx * dx + dy * dy, 1.5) + eps;
            float coeff = G * otherBody.m / dist3;
            ax += coeff * dx;
            ay += coeff * dy;
        }
        __syncthreads();
    }

    if(cursor)
    {
        float dx = rx - myBody.x;
        float dy = ry - myBody.y;
        float dist3 = pow(dx * dx + dy * dy, 1.5) + eps;
        // magnify the effect of cursor
        float coeff =  1e7 * G * cursor_weight / dist3;
        ax += coeff * dx;
        ay += coeff * dy;
    }

    myBody.vx += ax * dt;
    myBody.vy += ay * dt;
    myBody.x += myBody.vx * dt;
    myBody.y += myBody.vy * dt;
    // after collsision, the actual positon should
    // be symmetric correspoding to boundary
    // here we assume the velocity won't be too large,
    // so at most one collision can happen between two frames
    // otherwise we need a for loop here
    if(myBody.x > 1.0)
    {
        myBody.vx *= -collision_damping;
        // more precicsely, x = 1.6 - 0.6*x;
        myBody.x = 2.0 - myBody.x;
    }
    else if(myBody.x < -1.0)
    {
        myBody.vx *= -collision_damping;
        myBody.x = -2.0 - myBody.x;
    }
    if(myBody.y > 1.0)
    {
        myBody.vy *= -collision_damping;
        myBody.y = 2.0 -myBody.y;
    }
    else if(myBody.y < -1.0)
    {
        myBody.vy *= -collision_damping;
        myBody.y = -2.0 - myBody.y;
    }
    d_bodies[offset] = myBody;
}



void NBodyTimestepCuda(struct body* d_bodies, float rx, float ry, bool cursor)
{
    /**
    Compute a time step on the CUDA device.
    \param bodies A collection of bodies (located on the device).
    \param rx position x of the cursor.
    \param ry position y of the cursor.
    \param cursor Enable the mouse interaction if true (adding a weight = cursor_weight body in the computation).
    */
    float dt = 1e-3;
    int blockSize = 256;
    int gridSize =  (NUM_BODIES + blockSize - 1) / blockSize;
    unsigned int ssize = blockSize * sizeof(struct body);
    scalableUpdate<<<gridSize, blockSize, ssize>>>(d_bodies, dt, rx, ry, cursor, NUM_BODIES);
}
