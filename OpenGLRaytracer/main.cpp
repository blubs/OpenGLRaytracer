#include <stdio.h>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include "Utils.h"

#include <glm\gtc\matrix_transform.hpp>



void set_window_size_callback(GLFWwindow *win, int new_width, int new_height);
void setup_screen_texture();
void temp_update();



int window_width = 600;
int window_height = 400;
float window_aspect_ratio;

// Define the max window width and height
// Note: This is an arbitrary max width / height so we avoid 
// having to reallocate the screen texture after every window resize
#define MAX_WINDOW_WIDTH	3840
#define MAX_WINDOW_HEIGHT	2160



// The texture ID of the fullscreen texture
GLuint screen_texture_id;
// The screen texture RGB888 color data
unsigned char *screen_texture = nullptr;


// The vertex array object for the fullscreen quad
GLuint fullscreen_quad_vao;

// The vertex buffer object for the fullscreen quad
// [0] : the vertex position data
// [1] : the vertex texture coordinate data
GLuint fullscreen_quad_vbo[2];


GLuint screen_quad_shader = 0;
//TODO: need to get the locations of the vertex attributes

GLuint raytrace_compute_shader = 0;



// A temporary function where I can setup init code
// FIXME: remove this
void temp_init();

// A temporary function where I can setup update code
// FIXME: remove this
void temp_update();


int main(void)
{
	// Initializing glfw
	if (!glfwInit())
	{
		exit(EXIT_FAILURE);
	}

	// Specifying OpenGL version
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	// Creating the window
	GLFWwindow *window = glfwCreateWindow(window_width, window_height, "OpenGL Raytracer", NULL, NULL);

	glfwGetWindowSize(window, &window_width, &window_height);
	window_aspect_ratio = (float)window_width / (float)window_height;

	//TODO: handle window size change at runtime
	glfwSetWindowSizeCallback(window, set_window_size_callback);

	// Specifying this window as the current context
	glfwMakeContextCurrent(window);

	// Initializing glew
	if (glewInit() != GLEW_OK)
	{
		exit(EXIT_FAILURE);
	}

	// Specifying how long to wait before swapping screen buffers
	glfwSwapInterval(1);

	//============================
	temp_init();
	//============================

	// Continuously draw the screen until it should close
	while (!glfwWindowShouldClose(window))
	{
		//============================
		temp_update();
		//============================

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
	
	return 1;
}


// This callback is executed every time the window changes size
// This function is responsible for updating the fullscreen texture
void set_window_size_callback(GLFWwindow *win, int new_width, int new_height)
{
	window_width = new_width;
	window_height = new_height;
	window_aspect_ratio = (float)window_width / (float)window_height;

	glViewport(0, 0, window_width, window_height);

	// Resize the fullscreen texture:
	setup_screen_texture();

	// Draw the screen
	temp_update();
	glfwSwapBuffers(win);
}


void setup_screen_texture()
{
	// If the window dimensions are invalid, stop
	if (window_width <= 0 || window_height <= 0)
	{
		printf("Error: Invalid window dimensions - ( %d x %d ).\n", window_width, window_height);
		return;
	}

	// Don't let the internal window width / height exceed the max window width / height


	// If the texture memory has not been allocated, allocate it
	if (screen_texture == nullptr)
	{
		// Allocating the max amount of screen texture size that this program supports.
		screen_texture = (unsigned char*)malloc(sizeof(unsigned char) * 3 * MAX_WINDOW_WIDTH * MAX_WINDOW_HEIGHT);
	}

	// Wiping the texture memory that we are using
	//memset(screen_texture, 0, sizeof(char) * 3 * window_width * window_height);

	//=================================================================
	//TEMP - Filling the texture with a test gradient
	//=================================================================
	for (int i = 0; i < window_height; i++)
	{
		for (int j = 0; j < window_width; j++)
		{
			float r = 255 - (j % 255);
			float g = i % 255;
			float b = 255;

			// Adding a grid pattern every 10 pixels
			if (i % 10 == 0 || j % 10 == 0)
				r = g = b = 255;

			screen_texture[i * window_width * 3 + j * 3 + 0] = (char)r;
			screen_texture[i * window_width * 3 + j * 3 + 1] = (char)g;
			screen_texture[i * window_width * 3 + j * 3 + 2] = (char)b;
		}
	}
	//=================================================================

	// If the texture exists in OpenGL, delete it.
	if (glIsTexture(screen_texture_id) == GL_TRUE)
	{
		glDeleteTextures(1, &screen_texture_id);
	}
	// Create the OpenGL Texture
	glGenTextures(1, &screen_texture_id);
	printf("Creating a textur with ID: %d of size: %d x %d\n",screen_texture_id, window_width, window_height);

	glBindTexture(GL_TEXTURE_2D, screen_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, (const void *)screen_texture);
	glBindTexture(GL_TEXTURE_2D, 0);
}

// A temporary function where I can test out random code
// FIXME: remove this
void temp_init()
{
	//=======================================================
	// Creating Screen Image Texture
	//=======================================================
	setup_screen_texture();
	//=======================================================

	//=======================================================
	// Creating Fullscreen Quad Vertex Data
	//=======================================================
	// Defining the fullscreen quad vertices
	const float fullscreen_quad_verts[] =
	{
		-1.0f, 1.0f, 0.3f,
		-1.0f,-1.0f, 0.3f,
		1.0f, -1.0f, 0.3f,
		1.0f, -1.0f, 0.3f,
		1.0f,  1.0f, 0.3f,
		-1.0f,  1.0f, 0.3f
	};

	const float fullscreen_quad_uvs[] =
	{
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
	};

	// Creating the Vertex Array object
	glGenVertexArrays(1, &fullscreen_quad_vao);
	glBindVertexArray(fullscreen_quad_vao);
	glGenBuffers(2, fullscreen_quad_vbo);
	// Binding the vertex positions
	glBindBuffer(GL_ARRAY_BUFFER, fullscreen_quad_vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreen_quad_verts), fullscreen_quad_verts, GL_STATIC_DRAW);
	// Binding the vertex texture coordinates
	glBindBuffer(GL_ARRAY_BUFFER, fullscreen_quad_vbo[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreen_quad_uvs), fullscreen_quad_uvs, GL_STATIC_DRAW);
	//=======================================================

	//=======================================================
	// Creating Shaders
	//=======================================================

	screen_quad_shader = Utils::createShaderProgram("draw_screen_vert.glsl", "draw_screen_frag.glsl");

	// TODO: once I add compute shader source
	//raytrace_compute_shader = Utils::createShaderProgram("test.glsl");


}


// A temporary function where I can setup update code
// FIXME: remove this
void temp_update()
{
	//=======================================================
	// Calling the Raytrace shader
	//=======================================================


	//=======================================================
	// Drawing the result texture to the screen
	//=======================================================

	// Increment the color
	//static int color = 0;
	//color = ++color >= 3 ? 0 : color;

	/*switch (color)
	{
	case 0:
		glClearColor(1, 0, 0, 1);
		break;
	case 1:
		glClearColor(0, 1, 0, 1);
		break;
	case 2:
		glClearColor(0, 0, 1, 1);
		break;
	default:
		break;
	}*/
	glClear(0);


	// Don't draw the quad to the screen if we don't have the texture
	if (screen_texture == nullptr)
	{
		printf("Tried drawing with null texture - abort.\n");
		return;
	}

	glUseProgram(screen_quad_shader);


	// Binding the screen texture
	glActiveTexture(0);
	glBindTexture(GL_TEXTURE_2D, screen_texture_id);

	//glBindTexture(GL_TEXTURE_2D, texture_id);
	//GLint tex_loc = glGetUniformLocation(screen_quad_shader, "tex");
	//glUniform1i(tex_loc, 0);

	// Binding the vertex array object
	glBindVertexArray(fullscreen_quad_vao);

	// Binding the vertex position data
	glBindBuffer(GL_ARRAY_BUFFER, fullscreen_quad_vbo[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
	glEnableVertexAttribArray(0);
	// Binding the vertex texture coordinate data
	glBindBuffer(GL_ARRAY_BUFFER, fullscreen_quad_vbo[1]);
	glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, 0);
	glEnableVertexAttribArray(1);

	// Drawing the fullscreen quad
	//glDrawArrays(GL_LINE_STRIP, 0, 6);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

