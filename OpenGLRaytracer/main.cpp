#include <stdio.h>
#include <chrono>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>
#include "Utils.h"

#include <glm\gtc\matrix_transform.hpp>


void set_window_size_callback(GLFWwindow *win, int new_width, int new_height);
void draw();
void init();

// The resolution to render the raytrace at

#define RES_SCALE 1.0
#define RAYTRACE_RENDER_WIDTH	(int)(1280 * RES_SCALE)
#define RAYTRACE_RENDER_HEIGHT	(int)(720 * RES_SCALE)

int window_width = RAYTRACE_RENDER_WIDTH;//600
int window_height = RAYTRACE_RENDER_HEIGHT;//400


// The texture ID of the fullscreen texture
GLuint screen_texture_id;
// The screen texture RGBA8888 color data
unsigned char *screen_texture = nullptr;


// The vertex array object for the fullscreen quad
GLuint fullscreen_quad_vao;

// The vertex buffer object for the fullscreen quad
// [0] : the vertex position data
// [1] : the vertex texture coordinate data
GLuint fullscreen_quad_vbo[2];

GLuint screen_quad_shader = 0;
GLuint raytrace_compute_shader = 0;

// The time (in seconds) that the program began at
std::chrono::time_point<std::chrono::steady_clock> start_time;



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

	init();
	
	// Continuously draw the screen until it should close
	while (!glfwWindowShouldClose(window))
	{
		draw();
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

	glViewport(0, 0, window_width, window_height);

	draw();

	glfwSwapBuffers(win);
}

// Returns the current time in seconds
float get_current_time()
{
	auto cur_time = std::chrono::high_resolution_clock::now();

	long long elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - start_time).count();

	return elapsed_time / 1000.0f;
}



void init()
{
	//=======================================================
	// Creating Screen Image Texture
	//=======================================================

	// Allocating the memory for the screen texture
	screen_texture = (unsigned char*)malloc(sizeof(unsigned char) * 4 * RAYTRACE_RENDER_WIDTH * RAYTRACE_RENDER_HEIGHT);

	// Wiping the texture's memory contents
	memset(screen_texture, 0, sizeof(char) * 4 * RAYTRACE_RENDER_WIDTH * RAYTRACE_RENDER_HEIGHT);

	//==========================================
	// Setting the texture to pink
	// If on program execution, the texture is still pink, we know
	// there was an error in the raytrace compute shader
	//==========================================
	for (int i = 0; i < RAYTRACE_RENDER_HEIGHT; i++)
	{
		for (int j = 0; j < RAYTRACE_RENDER_WIDTH; j++)
		{
			screen_texture[i * RAYTRACE_RENDER_WIDTH * 4 + j * 4 + 0] = 250;
			screen_texture[i * RAYTRACE_RENDER_WIDTH * 4 + j * 4 + 1] = 128;
			screen_texture[i * RAYTRACE_RENDER_WIDTH * 4 + j * 4 + 2] = 255;
			screen_texture[i * RAYTRACE_RENDER_WIDTH * 4 + j * 4 + 3] = 255;
		}
	}
	//==========================================

	// Create the OpenGL Texture
	glGenTextures(1, &screen_texture_id);
	glBindTexture(GL_TEXTURE_2D, screen_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RAYTRACE_RENDER_WIDTH, RAYTRACE_RENDER_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, (const void *)screen_texture);
	glBindTexture(GL_TEXTURE_2D, 0);
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
	raytrace_compute_shader = Utils::createShaderProgram("raytrace_compute.glsl");

	// Storing the start time
	start_time = std::chrono::high_resolution_clock::now();
}


void draw()
{
	// Getting the current time:
	float cur_time = get_current_time();

	//=======================================================
	// Calling the Raytrace compute shader
	//=======================================================

	// Bind the raytrace compute shader
	glUseProgram(raytrace_compute_shader);

	// Bind the screen_texture_id texture to an image unit as the compute shader's output
	glBindImageTexture(0, screen_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

	// Bind the current time to a shader parameter
	glUniform1f(glGetUniformLocation(raytrace_compute_shader, "time"), cur_time);

	// Execute the raytrace shader in WIDTH X HEIGHT X 1 groups of size (1 x 1 x 1)
	int pixelsPerGroup = 1;

	int workGroupsX = RAYTRACE_RENDER_WIDTH;
	int workGroupsY = RAYTRACE_RENDER_HEIGHT;
	int workGroupsZ = 1;

	glDispatchCompute(workGroupsX, workGroupsY, workGroupsZ);

	// Wait until the compute shader has finished executing
	glFinish();

	//=======================================================
	// Drawing the result texture to the screen
	//=======================================================
	glClear(0);
	glUseProgram(screen_quad_shader);
	// Binding the screen texture
	glActiveTexture(0);
	glBindTexture(GL_TEXTURE_2D, screen_texture_id);
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
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

