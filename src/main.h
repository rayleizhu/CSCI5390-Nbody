#ifndef SAKUYA_H
#define SAKUYA_H


//Screen dimension constants
const int SCREEN_WIDTH = 480;
const int SCREEN_HEIGHT = 480;

// numbers of bodies in the simulation 
//const int NUM_BODIES = 8192;
const int NUM_BODIES = 4096;

// the body structure
struct body
{
	float x;	// the x position
	float y;	// the y position
	float vx;	// the x-axis velocity
	float vy;	// the y-axis velocity

	float m;	// the body mass
};


// function definitions for the main procedure
void rasterize(struct body* d_bodies, unsigned char* d_buffer, unsigned char* h_buffer);
void initializeNBodyCuda(struct body* &d_bodies, unsigned char* &d_buffer);
void freeMem(struct body* d_bodies, unsigned char* d_buffer);
void NBodyTimestepCuda(struct body* d_bodies, float rx, float ry, bool cursor);

#endif