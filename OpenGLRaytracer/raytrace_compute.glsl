#version 430

// The output texture
uniform image2D output_texture;

// Each group runs on one pixel
layout (local_size_x = 1, local_size_y = 1) in;

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

	//TODO: Do raytracing computations.
	imageStore(output_texture, pixel, vec4((pixel.x % 640) / 640.0,(pixel.y % 360) / 360.0,1.0,0.0));
}
