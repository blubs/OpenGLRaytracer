#version 430

// Time (seconds)
uniform float time;

// The output texture
uniform image2D output_texture;

// Each group runs on one pixel
layout (local_size_x = 1, local_size_y = 1) in;


struct Ray
{
	// The origin of the ray
	vec3 start;
	// The normalized direction of the ray
	vec3 dir;

	//TODO: 1/dir is used very often, we can introduce an optimization to have a function
	// that's used like r = computeRayValues(r);
	// which computes and stores 1/dir in the ray, then functions can just use pre-computed value instead of dividing each time.
};

struct Camera
{
	// The camera's world-space position
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


struct Box
{
	vec3 mins;
	vec3 maxs;
};

struct Object
{
	// We will check each of these in order
	//The box's axis-aligned bounding box defined relative to the object's center
	Box box;
	// The position of the Box_Object
	vec3 position;
	// The pitch, yaw, and roll of the object applied in that order (degrees)
	vec3 angles;
	// The surface color of the object
	vec3 color;
};


mat4 calc_projection_matrix(Camera c);
//mat4 calc_projection_matrix(float t, float r, float b, float l, float n, float f);
vec3 calc_view_ray(ivec2 pixel, int width, int height);
mat4 calc_view_matrix(Camera c);
vec3 raytrace(Ray r);

// Defining the null object types
Box null_box = { vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0) };


Object[] objects =
{
	{
		null_box,
		vec3( 0.0, 0.0, 0.0),
		vec3( 0.0, 0.0, 0.0),
		vec3( 1.0, 0.0, 0.0)
	},
	{
		{
			vec3(-1.0,-1.0,-1.0),
			vec3( 1.0, 1.0, 1.0)
		},
		// Make the box bob up and down
		vec3( 0.0, 0.0, sin(time * 3.0)),
		vec3( 0.0, 0.0, 0.0),
		vec3( 1.0, 0.0, 0.0)
	},
	{
		{
			vec3(-10.0,-10.0,-1.0),
			vec3( 10.0, 10.0, 1.0)
		},
		vec3( 0.0, 0.0,-3.0),
		// Make the box lean from one side to the other
		vec3( sin(time * 5.0) * 10.0, 45.0, 0.0),
		vec3( 0.0, 1.0, 0.0)
	},
	{
		{
			vec3(-1.0,-1.0,-2.0),
			vec3( 1.0, 1.0, 2.0)
		},
		vec3( 3.0, 4.0, 4.0),
		// Make the box tumble in the air
		vec3( 45.0 + time * 45.0, 0.0, 45.0 + time * 180.0),
		vec3( 0.0, 0.0, 1.0)
	},
};

int objects_count = 4;


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
	float radius = 10.0;
	float speed = time * 1.0;

	c.position = vec3(0.0, -radius, 0.0);
	// Make the camera orbit the origin
	c.position = vec3(radius * cos(speed), radius * sin(speed), 0);
	
	float pitch = 0.0;
	float yaw = 0.0;
	float roll = 0.0;
	
	// Enable this line to make the camera bob up and down
	//pitch = sin(speed * 2.0) * 10.0;
	// Enable this line to make the camera look at the center
	yaw = mod(1.0 * speed * (180.0/3.1416), 360.0) + 90.0;
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


	// Constructing the world-space-ray	
	Ray world_ray;
	world_ray.start = c.position;
	world_ray.dir = normalize((world_ray_end - world_ray_start).xyz);
	//========================================


	//========================================
	// Intersecting the ray with objects
	//========================================

	vec3 color = raytrace(world_ray);

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
	mat4 result = mat4(1.0);
	// First apply yaw (rotation about z-axis (up))
	result *= rotation_matrix_z(r.y);
	// Then apply the pitch (rotation about x-axis (right))
	result *= rotation_matrix_x(r.x);
	// Then apply the roll (rotation about y-axis (forward))
	result *= rotation_matrix_y(r.z);

	return result;
}

// Computes and returns a transformation matrix given position and angles
// pos : the translation of the transformation
// angles : the (pitch,yaw,roll) in degrees of the rotation of the transformation
mat4 calc_transform_matrix(vec3 position, vec3 angles)
{
	return  translation_matrix(position) * rotation_matrix(angles);
}

// Computes and returns the camera's current view matrix
// c : the camera for which to compute the view matrix
mat4 calc_view_matrix(Camera c)
{
	// I want the world-coordinate system to be a right-handed z-axis up coordinate system
	// so I will introduce a 90-degree rotation about the x-axis to convert from y-axis up to z-axis up
	mat4 flip_y_and_z = rotation_matrix_x(90.0);
	
	return inverse(calc_transform_matrix(c.position, c.angles) * flip_y_and_z);
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


vec2 intersect_box_object(Ray r, Object o)
{
	// Compute the box's local-space to world-space transform matrix:
	mat4 local_to_world = calc_transform_matrix(o.position,o.angles);
	// Invert it to get the box's world-space to local-space transform:
	mat4 world_to_local = inverse(local_to_world);

	// Convert the world-space ray to the box's local space:
	vec3 ray_start = (world_to_local * vec4(r.start,1.0)).xyz;
	vec3 ray_dir = (world_to_local * vec4(r.dir,0.0)).xyz;

	// Calculate the box's world mins and maxs:
	Box b = o.box;

	vec3 t_min = (b.mins - ray_start) / ray_dir;
	vec3 t_max = (b.maxs - ray_start) / ray_dir;
	vec3 t1 = min(t_min, t_max);
	vec3 t2 = max(t_min, t_max);
	float t_near = max(max(t1.x,t1.y),t1.z);
	float t_far = min(min(t2.x, t2.y), t2.z);

	return vec2(t_near, t_far);
}

vec3 raytrace(Ray r)
{
	float closest = 10000;
	int closest_index = -1;

	for(int i = 0; i < objects_count; i++)
	{
		float dist;
		
		// If the object is a box
		if(objects[i].box != null_box)
		{
			vec2 lambda = intersect_box_object(r, objects[i]);
			// If the ray intersects the box
			if(lambda.x > 0.0 && lambda.x < lambda.y)
			{
				dist = lambda.x;
			}
			else 
			{
				continue;
			}
		}
		// If the object is a sphere
		//else if(objects[i].sphere != null_sphere)
		//TODO 
		// etc...
		else
		{
			continue;
		}

		if(dist < closest)
		{
			closest = dist;
			closest_index = i;
		}
	}

	if(closest_index != -1)
	{
		return objects[closest_index].color;
	}
	return vec3(0.0,0.0,0.0);
}