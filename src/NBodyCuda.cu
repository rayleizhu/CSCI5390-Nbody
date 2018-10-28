#include "main.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>

#include "NBody.cuh"

#include <cuda.h>
#include <cuda_runtime.h>


double getRandom(double min, double max)
{
	double r = (double)rand() / RAND_MAX;
	return r*(max - min) + min;
}

void rasterize(struct body* bodies, unsigned char* buffer)
{
	/**
	rasterize the bodies from x,y: (-1,-1) to (1,1) according to some kind of formula

	Note: You can change the code for better visualization
	As the following code can be parallelized, you can optimize this routine with CUDA.

	\param bodies A collection of bodies (located on the device).
	\param buffer the RGB buffer for screen display (located on the host).
	*/

	// clear the canvas
	memset(buffer, 0, SCREEN_WIDTH*SCREEN_HEIGHT * 3 * sizeof(unsigned char));

	//TODO: copy the device memory to the host, and draw points on the canvas

	// Following is a sample of drawing a nice picture to the buffer.
	// You will know the index for each pixel.
	// The pixel value is from 0-255 so the data type is in unsigned char.

	//for (int x = 0; x < SCREEN_WIDTH; x++)
	//{
	//	for (int y = 0; y < SCREEN_HEIGHT; y++)
	//	{
	//		// the R channel
	//		buffer[x * SCREEN_WIDTH * 3 + y * 3 + 0] = (unsigned char)(x + y) | (unsigned char)(x - y);
	//		// the G channel
	//		buffer[x * SCREEN_WIDTH * 3 + y * 3 + 1] = (unsigned char)(sqrt((x-SCREEN_WIDTH/2)*(x - SCREEN_WIDTH / 2) + (y - SCREEN_HEIGHT / 2)*(y - SCREEN_HEIGHT / 2)) * 1.5);
	//		// the B channel
	//		buffer[x * SCREEN_WIDTH * 3 + y * 3 + 2] = (unsigned char)((x % 255) | (y % 255) );
	//	}
	//}
	int x, y;
	for (int i = 0; i < NUM_BODIES; i++)
    {
	    x = (int)((bodies[i].x + 1.0)/2.0 * SCREEN_WIDTH);
        y = (int)((bodies[i].y + 1.0)/2.0 * SCREEN_HEIGHT);
        buffer[x * SCREEN_HEIGHT * 3 + y * 3 + 0] = 255;
        buffer[x * SCREEN_HEIGHT * 3 + y * 3 + 1] = 255;
        buffer[x * SCREEN_HEIGHT * 3 + y * 3 + 2] = 255;
    }
}

struct body* initializeNBodyCuda()
{
	/**
	initialize the bodies, then copy to the CUDA device memory
	return the device pointer so that it can be reused in the NBodyTimestepCuda function.
	*/
	// initialize the position and velocity
	// you can implement own initial conditions to form a sprial/ellipse galaxy, have fun.
	struct body* bodies = new struct body[NUM_BODIES];
	for(int i = 0; i < NUM_BODIES; i++)
    {
		bodies[i].x = getRandom(-1.0, 1.0);
        bodies[i].y = getRandom(-1.0, 1.0);
        bodies[i].vx = getRandom(-1.0, 1.0);
        bodies[i].vy = getRandom(-1.0, 1.0);
        //bodies[i].vx = 10;
        //bodies[i].vy = 10;
        bodies[i].m = getRandom(1e5, 1e7);
    }
    return bodies;
}


void NBodyTimestepCuda(struct body* bodies, double rx, double ry, bool cursor)
{
	/**
	Compute a time step on the CUDA device. 
	TODO: correctly manage the device memory, compute the time step with proper block/threads 

	\param bodies A collection of bodies (located on the device).
	\param rx position x of the cursor.
	\param ry position y of the cursor.
	\param cursor Enable the mouse interaction if true (adding a weight = cursor_weight body in the computation).
	*/
	double delta_t = 1e-3;
	double a[NUM_BODIES][2];
	memset(a, 0, sizeof(a));
	// here we use the antisymmetry of [Fij] to save computation;
	for(int i = 0; i < NUM_BODIES; i++)
    {
	    for(int j = i+1; j < NUM_BODIES; j++)
	    {
            double xij = bodies[j].x - bodies[i].x;
            double yij = bodies[j].y - bodies[i].y;
            double dij3 = pow(pow(xij, 2) + pow(yij, 2), 1.5) + eps ;
            a[i][0] += G * bodies[j].m / dij3 * xij;
            a[i][1] += G * bodies[j].m / dij3 * yij;
            a[j][0] += -G * bodies[i].m / dij3 * xij;
            a[j][1] += -G * bodies[i].m / dij3 * yij;
        }
		if(cursor)
		{
			double xic = rx - bodies[i].x;
			double yic = ry - bodies[i].y;
			double dic3 = pow(pow(xic, 2) + pow(yic, 2), 1.5) + eps;
			a[i][0] += 1e7 * G * cursor_weight / dic3 * xic;
			a[i][1] += 1e7 * G * cursor_weight / dic3 * yic;
		}
        bodies[i].vx += delta_t * a[i][0];
	    bodies[i].vy += delta_t * a[i][1];
	    bodies[i].x += delta_t * bodies[i].vx;
	    bodies[i].y += delta_t * bodies[i].vy;
	    // consider collision
		// is weird is I use global collision factor, the effect
		// is as if  collision = 0.0
		// solved, because __constant__ variable can't be read dirctly by
		// host code
	    if(bodies[i].x < -1.0 && bodies[i].vx < 0)
        {
			bodies[i].vx = -collision_damping * bodies[i].vx;
			bodies[i].x = -2.0 - bodies[i].x;
        }
	    else if(bodies[i].x > 1.0 && bodies[i].vx > 0)
		{
			bodies[i].vx = -collision_damping * bodies[i].vx;
			bodies[i].x = 2.0 - bodies[i].x;
		}

	    if(bodies[i].y < -1.0 && bodies[i].vy < 0)
        {
			bodies[i].vy = -collision_damping * bodies[i].vy;
			bodies[i].y = -2.0 - bodies[i].y;
        }
	    else if(bodies[i].y > 1.0 && bodies[i].vy > 0)
		{
			bodies[i].vy = -collision_damping * bodies[i].vy;
			bodies[i].y = 2.0 - bodies[i].y;
		}
    }
}
