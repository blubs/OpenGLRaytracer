#include <stdio.h>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>


#include <glm\gtc\matrix_transform.hpp>



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
	GLFWwindow *window = glfwCreateWindow(600, 400, "OpenGL Raytracer", NULL, NULL);

	// Specifying this window as the current context
	glfwMakeContextCurrent(window);

	// Initializing glew
	if (glewInit() != GLEW_OK)
	{
		exit(EXIT_FAILURE);
	}

	// Specifying how long to wait before swapping screen buffers
	glfwSwapInterval(1);

	int color = 0;

	// Continuously draw the screen until it should close
	while (!glfwWindowShouldClose(window))
	{
		// Increment the color
		color = ++color > 3 ? 0 : color;

		switch (color)
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
		}

		//TODO: draw the window here
		glClear(0);


		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
	
	return 1;
}