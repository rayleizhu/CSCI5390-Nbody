#ifndef SAKUYA_H
#define SAKUYA_H


//Screen dimension constants
const int SCREEN_WIDTH = 480;
const int SCREEN_HEIGHT = 480;

// numbers of bodies in the simulation 
//const int NUM_BODIES = 8192;
const int NUM_BODIES = 1024;

// the body structure
struct body
{
	double x;	// the x position
	double y;	// the y position
	double vx;	// the x-axis velocity
	double vy;	// the y-axis velocity

	double m;	// the body mass
};


// function definitions for the main procedure
void rasterize(struct body* d_bodies, unsigned char* d_buffer, unsigned char* h_buffer);
void initializeNBodyCuda(struct body* &d_bodies, unsigned char* &d_buffer);
void freeMem(struct body* d_bodies, unsigned char* d_buffer);
void NBodyTimestepCuda(struct body* d_bodies, double rx, double ry, bool cursor);

#endif