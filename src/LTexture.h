#pragma once
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <string>


//Texture wrapper class
class LTexture
{
	public:
		//Initializes variables
		LTexture();

		//Deallocates memory
		~LTexture();

		// set the renderer
		void setRenderer(SDL_Renderer* renderer);

		// set the font for render
		void setFont(TTF_Font* font);

		//Loads image at specified path
		bool loadFromFile(std::string path);

		//Loads image from a memory buffer
		bool loadFromMemory(unsigned char* buffer);
		bool createFromBuffer(int width, int height, unsigned char* buffer);

		//Creates image from font string
		bool loadFromRenderedText(std::string textureText, SDL_Color textColor);

		//Deallocates texture
		void free();

		//Set color modulation
		void setColor(Uint8 red, Uint8 green, Uint8 blue);

		//Set blending
		void setBlendMode(SDL_BlendMode blending);

		//Set alpha modulation
		void setAlpha(Uint8 alpha);

		//Renders texture at given point
		void render(int x, int y, SDL_Rect* clip = NULL, double angle = 0.0, SDL_Point* center = NULL, SDL_RendererFlip flip = SDL_FLIP_NONE);

		//Gets image dimensions
		int getWidth();
		int getHeight();

	private:
		//The actual hardware texture
		SDL_Texture* mTexture;

		//Image dimensions
		int mWidth;
		int mHeight;

		SDL_Renderer *renderer;
		TTF_Font *font;
};