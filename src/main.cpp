#include "SDL2/SDL.h"
#include <cstdio>
#include <string>
#include <sstream>
#include <time.h>
#include <cuda_runtime_api.h>

#include "main.h"
#include "LTexture.h"


int main(int argc, char *argv[])
{
	// Initialize PRNG
	srand((unsigned int)time(NULL));

	// Define common variables for SDL rendering
	SDL_Window *window = NULL;
	SDL_Renderer *renderer = NULL;

	//Initialize SDL
	bool success = true;
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL Error: %s\n", SDL_GetError());
		success = false;
	}
	else
	{
		//Set texture filtering to linear
		if (!SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1"))
		{
			printf("Warning: Linear texture filtering not enabled!");
		}

		//Create window
		window = SDL_CreateWindow("N-Body Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
		if (window == NULL)
		{
			printf("Window could not be created! SDL Error: %s\n", SDL_GetError());
			success = false;
		}
		else
		{
			//Create vsynced renderer for window
			renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
			if (renderer == NULL)
			{
				printf("Renderer could not be created! SDL Error: %s\n", SDL_GetError());
				success = false;
			}
			else
			{
				//Initialize renderer color
				SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
			}
			if (TTF_Init() == -1)
			{
				printf("SDL_ttf could not initialize! SDL_ttf Error: %s\n", TTF_GetError());
				success = false;
			}
		}
	}

	// Begins to build the rendering
	if (!success)
	{

	}
	else
	{
		// prepare the materials for rendering
		// the main canvas (reading from memory)
		LTexture canvasTexture;
		canvasTexture.setRenderer(renderer);
		unsigned char canvas[SCREEN_WIDTH*SCREEN_HEIGHT * 3];
		memset(canvas, 0, SCREEN_WIDTH*SCREEN_HEIGHT * 3 * sizeof(unsigned char));

		// initialize N-bodies
		struct body *bodies = initializeNBodyCuda();

		
		// the FPS indicator
		LTexture gTextTexture;
		TTF_Font *gFont = TTF_OpenFont("lazy.ttf", 28);
		gTextTexture.setFont(gFont);
		gTextTexture.setRenderer(renderer);
		if (gFont == NULL)
		{
			printf("Failed to load lazy font! SDL_ttf Error: %s\n", TTF_GetError());
			success = false;
		}
		std::stringstream timeText;
		Uint32 startTime = SDL_GetTicks();
		int frameCount = 0;
		int fps = 0;

		// check mouse down
		bool cursor = false;

		//Main loop flag
		bool quit = false;

		//Event handler
		SDL_Event e;

		//While application is running
		while (!quit)
		{

			//Handle events on queue
			while (SDL_PollEvent(&e) != 0)
			{
				//User requests quit
				if (e.type == SDL_QUIT)
				{
					quit = true;
				}
				if (e.type == SDL_MOUSEBUTTONDOWN)
				{
					cursor = true;
				}
				if (e.type == SDL_MOUSEBUTTONUP)
				{
					cursor = false;
				}
			}

			//Clear screen
			SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
			SDL_RenderClear(renderer);

			// Retrieve cursor position, normalize to real space
			int x, y;
			SDL_GetMouseState(&x, &y);
			double rx, ry;
			rx = (double)(y) / (double)(SCREEN_WIDTH);
			ry = (double)(x) / (double)(SCREEN_HEIGHT);
			rx = rx * 2 - 1;
			ry = ry * 2 - 1;

			NBodyTimestepCuda(bodies,rx,ry,cursor);
			rasterize(bodies, canvas);

			canvasTexture.createFromBuffer(SCREEN_WIDTH, SCREEN_HEIGHT, canvas);
			canvasTexture.render(0, 0);


			// set the FPS counter
			SDL_Color textColor = { 255, 0 , 0};

			if (SDL_GetTicks() - startTime > 500)
			{
				fps = frameCount * 2;

				// cleanup
				frameCount = 0;
				startTime = SDL_GetTicks();
			}

			timeText.str("");
			timeText << "FPS: " << fps;
			gTextTexture.loadFromRenderedText(timeText.str().c_str(), textColor);
			gTextTexture.render(0, 0);


			//Update screen
			SDL_RenderPresent(renderer);

			frameCount++;
		}
		freeMem(bodies);
	}


	return 0;
}