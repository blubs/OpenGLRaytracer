#version 430

// Time (seconds)
uniform float time;

// The output texture
uniform image2D output_texture;

// Each group runs on one pixel
layout (local_size_x = 1, local_size_y = 1) in;


struct Camera
{
	vec3 position;
	// The pitch, yaw, and roll of the camera applied in that order (degrees)
	vec3 angles;
	// The camera's vertical fov (degrees)
	float v_fov;
	// The camera's aspect-ratio (width / height)
	float aspect;
	// The camera's near clipping plane
	float near;
	// The camera's far clipping plane
	float far;
};

mat4 calc_projection_matrix(Camera c);
//mat4 calc_projection_matrix(float t, float r, float b, float l, float n, float f);
vec3 calc_view_ray(ivec2 pixel, int width, int height);
mat4 calc_view_matrix(Camera c);
vec3 raytrace(vec3 org, vec3 dir);



void main()
{
	int width = int(gl_NumWorkGroups.x);
	int height = int(gl_NumWorkGroups.y);
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

	//========================================
	// Defining the view's camera
	//========================================
	Camera c;
	c.position = vec3(0,0,5);
	c.angles = vec3(0,0,0);

	// Making the camera orbit the origin
	float radius = 6.0;
	float speed = time * 2.0;
	// Make the camera orbit the origin
	c.position = vec3(radius * sin(speed), 0, radius * cos(speed));
	
	float pitch = 0.0;
	float yaw = 0.0;
	float roll = 0.0;
	
	// Enable this line to make the camera bob up and down
	//pitch = sin(speed * 2.0) * 10.0;
	// Enable this line to make the camera look at the center
	yaw = mod(1.0 * speed * (180.0/3.1416), 360.0);
	// Enable this line to make the camera roll around
	//roll = mod(0.5 * speed * (180.0/3.1416), 360.0);
	c.angles = vec3(pitch,yaw,roll);

	
	c.near = 0.1;
	c.far = 1000;
	c.aspect = 16.0/9.0;
	c.v_fov = 90.0;

	mat4 proj_mat = calc_projection_matrix(c);
	mat4 view_mat = calc_view_matrix(c);

	//========================================


	//========================================
	// Getting this pixel's world-space ray
	//========================================

	// Constructing the ray in view-space
	float view_ray_x = float(pixel.x - width/2) / float(width/2);
	float view_ray_y = float(pixel.y - height/2) / float(height/2);
	vec4 view_ray_start = vec4( view_ray_x, view_ray_y, 0.5, 1.0);
	vec4 view_ray_end = vec4(view_ray_x, view_ray_y, 1.0, 1.0);

	// Converting the ray to world-space
	mat4 inverse_proj_mat = inverse(proj_mat * view_mat);
	vec4 world_ray_start = inverse_proj_mat * view_ray_start;
	world_ray_start /= world_ray_start.w;
	vec4 world_ray_end = inverse_proj_mat * view_ray_end;
	world_ray_end /= world_ray_end.w;

	// for derivation:
	//NOTE: https://www.mathworks.com/matlabcentral/answers/373545-how-can-i-convert-from-the-pixel-position-in-an-image-to-3d-world-coordinates
	// The above code yields the same as this:
	//vec4 world_ray_v4 = inverse_proj_mat * view_ray;
	//world_ray_v4 /= world_ray_v4.w;
	//world_ray_v4 -= vec4(c.position,0.0);
	
	
	//vec3 world_ray = world_ray_v4.xyz;
	vec3 world_ray = (world_ray_end - world_ray_start).xyz;
	//========================================


	//========================================
	// Intersecting the ray with objects
	//========================================

	vec3 color = raytrace(c.position, world_ray);

	//TODO: intersect this ray with objects
	//TODO: Do raytracing computations.

	//FIXME: I'm getting an all-black screen
	// I should see a white cube in the middle of the screen
	//========================================

	vec3 final_color = color;

	imageStore(output_texture, pixel, vec4(final_color,0.0));
}

// Computes and returns a perspective projection matrix
// c : the camera for which to compute the projection matrix
mat4 calc_projection_matrix(Camera c)
{
	float deg_to_rad = 3.14159265358 / 180.0;

	float q = 1.0 / tan(deg_to_rad * 0.5 * c.v_fov);
	float A = q / c.aspect;
	float B = (c.near + c.far) / (c.near - c.far);
	float C = (2.0 * c.near * c.far) / (c.near - c.far);

	mat4 result = mat4(0.0);
	result[0][0] = A;
	result[1][1] = q;
	result[2][2] = B;
	result[2][3] = -1.0f;
	result[3][2] = C;

	return result;
}

// Computes a translation matrix given a 3D translation
// t : the translation for which to compute the matrix
mat4 translation_matrix(vec3 t)
{
	mat4 result = mat4(1.0);
	result[3] = vec4(t,1.0);
	return result;
}


// Computes a rotation matrix about the x-axis
// deg : the degrees to rotate by
mat4 rotation_matrix_x(float deg)
{
	float deg_to_rad = 3.14159265358 / 180.0;
	float cos_deg = cos(deg_to_rad * deg);
	float sin_deg = sin(deg_to_rad * deg);
	mat4 result = mat4(1.0);
	result[1][1] = cos_deg;
	result[1][2] = sin_deg;
	result[2][1] = -sin_deg;
	result[2][2] = cos_deg;
	return result;
}

// Computes a rotation matrix about the y-axis
// deg : the degrees to rotate by
mat4 rotation_matrix_y(float deg)
{
	float deg_to_rad = 3.14159265358 / 180.0;
	float cos_deg = cos(deg_to_rad * deg);
	float sin_deg = sin(deg_to_rad * deg);
	mat4 result = mat4(1.0);
	result[0][0] = cos_deg;
	result[0][2] = -sin_deg;
	result[2][0] = sin_deg;
	result[2][2] = cos_deg;
	return result;
}

// Computes a rotation matrix about the z-axis
// deg : the degrees to rotate by
mat4 rotation_matrix_z(float deg)
{
	float deg_to_rad = 3.14159265358 / 180.0;
	float cos_deg = cos(deg_to_rad * deg);
	float sin_deg = sin(deg_to_rad * deg);
	mat4 result = mat4(1.0);
	result[0][0] = cos_deg;
	result[0][1] = sin_deg;
	result[1][0] = -sin_deg;
	result[1][1] = cos_deg;
	return result;
}

// Computes a rotation matrix given a 3D rotation r
// r : the rotation in pitch, yaw, and roll
mat4 rotation_matrix(vec3 r)
{
	float deg_to_rad = 3.14159265358 / 180.0;
	mat4 result = mat4(1.0);
	// First apply yaw
	result *= rotation_matrix_y(r.y);
	// Then apply the pitch
	result *= rotation_matrix_x(r.x);
	// Then apply the roll
	result *= rotation_matrix_z(r.z);

	return result;
}

// Computes and returns the camera's current view matrix
// c : the camera for which to compute the view matrix
mat4 calc_view_matrix(Camera c)
{
	mat4 result = mat4(1.0);
	result *= rotation_matrix(c.angles);
	result *= translation_matrix(c.position);
	return inverse(result);
}

// Computes and returns a normalized view-space ray from the origin for a given pixel
// pixel : the pixel's x / y index
// width : the width of the image
// height : the height of the image
vec3 calc_view_ray(ivec2 pixel, int width, int height)
{
	float x = float(pixel.x - width/2)/float(width);
	float y = float(pixel.y - height/2)/float(height);

	return normalize(vec3( x, 1.0, y));
}

vec2 intersect_box(vec3 start, vec3 dir, vec3 box_mins, vec3 box_maxs)
{
	vec3 t_min = (box_mins - start) / dir;
	vec3 t_max = (box_maxs - start) / dir;
	vec3 t1 = min(t_min, t_max);
	vec3 t2 = max(t_min, t_max);
	float t_near = max(max(t1.x,t1.y),t1.z);
	float t_far = min(min(t2.x, t2.y), t2.z);
	return vec2(t_near, t_far);
}

vec3 raytrace(vec3 start, vec3 dir)
{
	float closest = 10000;

	vec3 box_mins = vec3(-1.0,-1.0,-1.0);
	vec3 box_maxs = vec3(1.0,1.0,1.0);
	vec2 lambda = intersect_box(start, dir, box_mins, box_maxs);

	if(lambda.x > 0.0 && lambda.x < lambda.y && lambda.x < closest)
		return vec3(1.0,1.0,1.0);

	//TODO: find the smallest value of lambda.x, that is what we're interesecting

	return vec3(0.0,0.0,0.0);
}