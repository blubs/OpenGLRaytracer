#version 430

// Time (seconds)
uniform float time;

// The output texture
uniform image2D output_texture;

// Each group runs on one pixel
layout (local_size_x = 1, local_size_y = 1) in;

// Math Constants
const float PI = 3.14159265358;
const float HALF_PI = PI * 0.5;
const float TWO_PI = PI * 2;
const float ROOT_TWO_PI = sqrt(TWO_PI);
const float DEG_TO_RAD = PI / 180.0;
const float RAD_TO_DEG = 180.0 / PI;


// How many levels deep the recursive raytrace is allowed to go
const int MAX_RAYTRACE_DEPTH = 0;


struct Ray
{
	// The origin of the ray
	vec3 start;
	// The normalized direction of the ray
	vec3 dir;
};

// Defining the null ray type
Ray null_ray = {vec3(0.0), vec3(0.0)};

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


//=============================================================================
//		Material Defs
//=============================================================================
struct Material
{
	// Phong-Shading properties
	vec4 ambient;			// Ambient Color
	vec4 diffuse;			// Diffuse Color
	vec4 specular;			// Specular Color
	float shininess;		// Phong-Shading specular exponent
	vec4 emissive;			// Emissive Color

	// Raytracing Material Properties
	float reflectivity;		// Strength of the Reflected Ray's color
	float transparency;		// Strength of the Refracted Ray's color
	float refraction_index;	// Index of refraction (1.0 for air, 1.5 for glass, should be > 1.0)
};

// The index of refraction of open space
const float air_index_of_refraction = 1.0;

Material material1 =
{
	vec4(1.0),				// Ambient Color
	vec4(0.5,0.0,0.0,1.0),	// Diffuse Color
	vec4(1.0),				// Specular Color
	4.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	1.0,					// Strength of the Reflected Ray's color
	0.0,					// Strength of the Refracted Ray's color
	1.5						// Index of refraction (1.5 = glass)
};

Material material2 =
{
	vec4(1.0),				// Ambient Color
	vec4(0.3,0.6,0.3,1.0),	// Diffuse Color
	vec4(1.0),				// Specular Color
	4.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	1.0,					// Strength of the Reflected Ray's color
	0.0,					// Strength of the Refracted Ray's color
	1.5						// Index of refraction (1.5 = glass)
};

Material red_glass_material =
{
	vec4(1.0),				// Ambient Color
	vec4(1.0,0.0,0.0,1.0),	// Diffuse Color
	vec4(1.0),				// Specular Color
	10.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	0.8,					// Strength of the Reflected Ray's color
	0.4,					// Strength of the Refracted Ray's color
	1.5						// Index of refraction (1.5 = glass)
};

Material green_glass_material =
{
	vec4(1.0),				// Ambient Color
	vec4(0.0,1.0,0.0,1.0),	// Diffuse Color
	vec4(1.0),				// Specular Color
	10.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	0.4,					// Strength of the Reflected Ray's color
	0.6,					// Strength of the Refracted Ray's color
	1.5						// Index of refraction (1.5 = glass)
};

Material blue_glass_material =
{
	vec4(1.0),				// Ambient Color
	vec4(0.0,0.0,1.0,1.0),	// Diffuse Color
	vec4(1.0),				// Specular Color
	10.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	0.4,					// Strength of the Reflected Ray's color
	0.6,					// Strength of the Refracted Ray's color
	1.5						// Index of refraction (1.5 = glass)
};


Material mirror_material =
{
	vec4(1.0),				// Ambient Color
	vec4(0.6,0.6,0.6,1.0),	// Diffuse Color
	vec4(1.0),				// Specular Color
	4.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	1.0,					// Strength of the Reflected Ray's color
	0.0,					// Strength of the Refracted Ray's colorgimp
	1.0						// Index of refraction
};

Material wall_material =
{
	vec4(0.5),				// Ambient Color
	vec4(0.4),				// Diffuse Color
	vec4(0.3),				// Specular Color
	3.0,					// Specular exponent
	vec4(0.0),				// Emissive Color
	0.3,					// Strength of the Reflected Ray's color
	0.0,					// Strength of the Refracted Ray's color
	1.0						// Index of refraction
};

//=============================================================================


//=============================================================================
//		Object Collision Definitions
//=============================================================================

struct Box
{
	vec3 mins;
	vec3 maxs;
};

struct Sphere
{
	float radius;
};

// Defining the null object types
Box null_box = { vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0) };
Sphere null_sphere = { -1.0 };

//=============================================================================


//=============================================================================
//		Light Definitions
//=============================================================================


// Phong-Shading Point Lights
struct Light
{
	vec3 position;				// The world position of the point light
	vec4 ambient;				// Ambient Color
	vec4 diffuse;				// Diffuse Color
	vec4 specular;				// Specular Color
};

// List of Phong-shading Point lights
Light[] lights =
{
	// World Ambient Light
	{
		vec3(0.1),				// Position
		vec4(0.3),				// Ambient Color
		vec4(0.0),				// Diffuse Color
		vec4(0.0)				// Specular Color
	},
	// Point Light # 1
	{
		vec3(7.0, 7.0, 2.0),	// Position
		vec4(0.05),				// Ambient Color
		vec4(1.0),				// Diffuse Color
		vec4(1.0)				// Specular Color
	},
	// Point Light # 2
	{
		vec3(3.0, -3.0, 4.0),	// Position
		vec4(0.05),				// Ambient Color
		vec4(1.0,0.0,0.0,1.0),	// Diffuse Color
		vec4(1.0,0.0,0.0,1.0)	// Specular Color
	}
};

int lights_count = 3;

//=============================================================================

// Forward declarations
mat4 calc_projection_matrix(Camera c);
vec3 calc_view_ray(ivec2 pixel, int width, int height);
mat4 calc_view_matrix(Camera c);
vec3 recursive_raytrace(Ray r, int max_depth);


// time_scale allows us to slow all time-dependent movements in the scene
float time_scale = 0.4;
float scaled_time = time * time_scale;


//=============================================================================
//		Object Definitions
//=============================================================================

struct Object
{
	// We will check each of these in order
	// An object should not have both a Box and a Sphere
	//The object's bounding box defined relative to the object's center
	Box box;
	// The object's sphere
	Sphere sphere;
	// The position of the Box_Object
	vec3 position;
	// The pitch, yaw, and roll of the object applied in that order (degrees)
	vec3 angles;
	// The material of the object
	Material material;
};


Object[] objects =
{
	// Large cube surrounding the scene
	{
		{
			vec3(-11.0),													// Bounding Box Mins
			vec3( 11.0)														// Bounding Box Maxs
		},
		null_sphere,														// Bounding Sphere
		vec3( 0.0),															// Position
		vec3( 0.0),															// Rotation
		wall_material														// Material
	},
	// Small red box in the center
	{
		{
			vec3(-1.0,-1.0,-1.0) * (0.5*sin(scaled_time * 0.5) + 1.5),		// Bounding Box Mins
			vec3( 1.0, 1.0, 1.0) * (0.5*sin(scaled_time * 0.5) + 1.5)		// Bounding Box Maxs
		},
		null_sphere,														// Bounding Sphere
		// Make the box bob up and down
		vec3( 0.0, 0.0, sin(scaled_time * 3.0)),							// Position
		vec3( 0.0, scaled_time * 90.0, 0.0),								// Rotation
		mirror_material														// Material
	},
	// Large green box the makes up the floor
	{
		{
			vec3(-10.0,-10.0,-1.0),											// Bounding Box Mins
			vec3( 10.0, 10.0, 1.0)											// Bounding Box Maxs
		},
		null_sphere,														// Bounding Sphere
		vec3( 0.0, 0.0,-3.0),												// Position
		// Make the box lean from one side to the other
		vec3( sin(scaled_time * 5.0) * 10.0, 45.0, 0.0),					// Rotation
		green_glass_material												// Material
	},
	// Tumbling blue box
	{
		{
			vec3(-1.0,-1.0,-2.0),											// Bounding Box Mins
			vec3( 1.0, 1.0, 2.0)											// Bounding Box Maxs
		},
		null_sphere,														// Bounding Sphere
		vec3( 3.0, 4.0, 1.0),												// Position (3,4,4)
		// Make the box tumble in the air
		vec3( 45.0 + scaled_time * 45.0, 0.0, 45.0 + scaled_time * 180.0),	// Rotation
		blue_glass_material													// Material
	},
	// Small blue sphere
	{
		null_box,															// Bounding Box
		{
			2.0																// Bounding Sphere Radius
		},
		vec3( -3.0, 4.0, 1.0),												// Position (3,4,4)
		vec3( 0.0),															// Rotation
		red_glass_material													// Material
	}
};
int objects_count = 5;

//=============================================================================

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

	c.position = vec3(0.0, -radius, 0.0);
	// Move the camera with time
	float speed = time * time_scale + 0.5;
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

	// Apply the camera angles
	c.angles = vec3(pitch,yaw,roll);

	// Set up the camera's view frustum
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

	// Calculating the ray in view-space
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
	// Casting the ray out into the world
	// and Intersecting the ray with objects
	//========================================

	vec3 final_color = recursive_raytrace(world_ray, MAX_RAYTRACE_DEPTH);

	// Write the final color to the output image pixel
	imageStore(output_texture, pixel, vec4(final_color,0.0));
}

//------------------------------------------------------------------------------
// Computes and returns a perspective projection matrix
// c : the camera for which to compute the projection matrix
//------------------------------------------------------------------------------
mat4 calc_projection_matrix(Camera c)
{
	float q = 1.0 / tan(DEG_TO_RAD * 0.5 * c.v_fov);
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

//------------------------------------------------------------------------------
// Computes a translation matrix given a 3D translation
// t : the translation for which to compute the matrix
//------------------------------------------------------------------------------
mat4 translation_matrix(vec3 t)
{
	mat4 result = mat4(1.0);
	result[3] = vec4(t,1.0);
	return result;
}


//------------------------------------------------------------------------------
// Computes a rotation matrix about the x-axis
// deg : the degrees to rotate by
//------------------------------------------------------------------------------
mat4 rotation_matrix_x(float deg)
{
	float cos_deg = cos(DEG_TO_RAD * deg);
	float sin_deg = sin(DEG_TO_RAD * deg);
	mat4 result = mat4(1.0);
	result[1][1] = cos_deg;
	result[1][2] = sin_deg;
	result[2][1] = -sin_deg;
	result[2][2] = cos_deg;
	return result;
}

//------------------------------------------------------------------------------
// Computes a rotation matrix about the y-axis
// deg : the degrees to rotate by
//------------------------------------------------------------------------------
mat4 rotation_matrix_y(float deg)
{
	float cos_deg = cos(DEG_TO_RAD * deg);
	float sin_deg = sin(DEG_TO_RAD * deg);
	mat4 result = mat4(1.0);
	result[0][0] = cos_deg;
	result[0][2] = -sin_deg;
	result[2][0] = sin_deg;
	result[2][2] = cos_deg;
	return result;
}

//------------------------------------------------------------------------------
// Computes a rotation matrix about the z-axis
// deg : the degrees to rotate by
//------------------------------------------------------------------------------
mat4 rotation_matrix_z(float deg)
{
	float cos_deg = cos(DEG_TO_RAD * deg);
	float sin_deg = sin(DEG_TO_RAD * deg);
	mat4 result = mat4(1.0);
	result[0][0] = cos_deg;
	result[0][1] = sin_deg;
	result[1][0] = -sin_deg;
	result[1][1] = cos_deg;
	return result;
}

//------------------------------------------------------------------------------
// Computes a rotation matrix given a 3D rotation r
// r : the rotation in pitch, yaw, and roll
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// Computes a rotation matrix given a 3D rotation
// angle radians about an arbitrary axis
// angle : the rotation in radians
// axis : the axis of rotation
// Implementation found on:
// http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
//------------------------------------------------------------------------------
mat3 rotation_matrix(vec3 axis, float angle)
{
	float s = -sin(angle);
	float c = cos(angle);
	float oc = 1.0 - c;
	vec3 as = axis * s;
	mat3 p = mat3(axis.x * axis, axis.y * axis, axis.z * axis);
	mat3 q = mat3(c, -as.z, as.y, as.z, c, -as.x, -as.y, as.x, c);
	return p * oc + q;
}

//------------------------------------------------------------------------------
// Computes and returns a transformation matrix given position and angles
// pos : the translation of the transformation
// angles : the (pitch,yaw,roll) in degrees of the rotation of the transformation
//------------------------------------------------------------------------------
mat4 calc_transform_matrix(vec3 position, vec3 angles)
{
	return  translation_matrix(position) * rotation_matrix(angles);
}

//------------------------------------------------------------------------------
// Computes and returns the camera's current view matrix
// c : the camera for which to compute the view matrix
//------------------------------------------------------------------------------
mat4 calc_view_matrix(Camera c)
{
	// I want the world-coordinate system to be a right-handed z-axis up coordinate system
	// so I will introduce a 90-degree rotation about the x-axis to convert from y-axis up to z-axis up
	mat4 flip_y_and_z = rotation_matrix_x(90.0);

	return inverse(calc_transform_matrix(c.position, c.angles) * flip_y_and_z);
}

//------------------------------------------------------------------------------
// Computes and returns a normalized view-space ray from the origin for a given pixel
// pixel : the pixel's x / y index
// width : the width of the image
// height : the height of the image
//------------------------------------------------------------------------------
vec3 calc_view_ray(ivec2 pixel, int width, int height)
{
	float x = float(pixel.x - width/2)/float(width);
	float y = float(pixel.y - height/2)/float(height);

	return normalize(vec3( x, 1.0, y));
}

struct Collision
{
	// The t-value at which this collision occurs for a ray
	float t;
	// The world position of the collision
	vec3 p;
	// The normal of the collision
	vec3 n;
	// Whether the collision occurs inside of the object
	bool inside;
	// The index of the object this collision hit
	int object_index;
};

// This defines what a null collision looks like
Collision null_collision = { -1.0, vec3(0.0), vec3(0.0), false, -1};

//------------------------------------------------------------------------------
// Checks if Ray r intersects the Sphere defined by Object o.sphere
// This implementation is based on the following algorithm:
// http://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/basic_algo.pdf
//------------------------------------------------------------------------------
Collision intersect_sphere_object(Ray r, Object o)
{
	float radius = o.sphere.radius;
	float qa = dot(r.dir, r.dir);
	float qb = dot( 2 * r.dir, r.start - o.position);
	float qc = dot( r.start - o.position, r.start - o.position) - radius * radius;

	// Solving for qa * t^2 + qb * t + qc = 0
	float qd = qb * qb - 4 * qa * qc;

	Collision c;
	c.inside = false;

	// If qd < 0, no solution
	if(qd < 0.0)
	{
		c.t = -1.0;
		return c;
	}

	float sqrt_qd = sqrt(qd);

	float t1 = (-qb + sqrt_qd) / (2.0 * qa);
	float t2 = (-qb - sqrt_qd) / (2.0 * qa);

	float t_near = min(t1, t2);
	float t_far = max(t1, t2);

	c.t = t_near;

	// If t_far < 0, the sphere is behind the ray, no intersection
	if(t_far < 0.0)
	{
		c.t = -1.0;
		return c;
	}

	// t_near < 0, the ray started inside the sphere
	if(t_near < 0.0)
	{
		c.t = t_far;
		c.inside = true;
	}

	// Compute the world position of the collision
	c.p = r.start + c.t * r.dir;

	// Use the world position to compute the surface normal
	c.n = normalize(c.p - o.position);

	// If the collision is leaving the sphere, flip the normal
	if(c.inside)
	{
		c.n *= -1.0;
	}

	return c;
}

//------------------------------------------------------------------------------
// Checks if Ray r intersects the Box defined by Object o.box
// This implementation is based on the following algorithm:
// http://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/basic_algo.pdf
//------------------------------------------------------------------------------
Collision intersect_box_object(Ray r, Object o)
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

	Collision c;
	c.t = t_near;
	c.inside = false;

	// If the ray is entering the box
	// t_near contains the farthest boundary of entry
	// If the ray is leaving the box
	// t_far contains the closest boundary of exit
	// The ray intersects the box if and only if t_near < t_far
	// and if t_far > 0.0

	// If the ray didn't intersect the box, return a negative t value
	if(t_near >= t_far || t_far <= 0.0)
	{
		c.t = -1.0;
		return c;
	}

	float intersection = t_near;
	vec3 boundary = t1;

	// if t_near < 0, then the ray started inside the box and left the box
	if( t_near < 0.0)
	{
		c.t = t_far;
		intersection = t_far;
		boundary = t2;
		c.inside = true;
	}

	// Checking which boundary the intersection lies on
	int face_index = 0;

	if(intersection == boundary.y)
		face_index = 1;
	else if(intersection == boundary.z)
		face_index = 2;

	// Creating the collision normal
	c.n = vec3(0.0);
	c.n[face_index] = 1.0;


	// If we hit the box from the negative axis, invert the normal
	if(ray_dir[face_index] > 0.0)
	{
		c.n *= -1.0;
	}

	// Converting the normal to world-space
	c.n = transpose(inverse(mat3(local_to_world))) * c.n;

	// Calculate the world-position of the intersection:
	c.p = (local_to_world * vec4(ray_start + c.t * ray_dir,1.0)).xyz;

	return c;
}

//------------------------------------------------------------------------------
// Returns the closest collision of a ray
//
// Returns a collision with a object_index of -1 if no collision
//
// A Collision struct consists of:
// t : the t-value of the collision along the ray
// p : the world-space position of the collision
// n : the world-space surface normal at the collision
// inside : whether this ray started inside and exited the object
// object_index : the index in the objects[] array of the object this collision hit
//------------------------------------------------------------------------------
Collision get_closest_collision(Ray r)
{
	float closest = 10000;
	Collision closest_collision;
	closest_collision.object_index = -1;

	for(int i = 0; i < objects_count; i++)
	{
		Collision c;

		// If the object is a box
		if(objects[i].box != null_box)
		{
			// If the ray intersects the box
			c = intersect_box_object(r, objects[i]);
			if(c.t <= 0.0)
			{
				continue;
			}
		}
		// If the object is a sphere
		else if(objects[i].sphere != null_sphere)
		{
			// If the ray intersects the sphere
			c = intersect_sphere_object(r, objects[i]);
			if(c.t <= 0.0)
			{
				continue;
			}
		}
		else
		{
			continue;
		}

		if(c.t < closest)
		{
			closest = c.t;
			closest_collision = c;
			closest_collision.object_index = i;
		}
	}

	return closest_collision;
}

//------------------------------------------------------------------------------
// Computes the Ambient Diffuse Specular (ADS) Phong lighting for an
// incident Ray r at a surface defined by Collision c
// Returns the color of the surface
//------------------------------------------------------------------------------
vec3 ads_phong_lighting(Ray r, Collision c)
{
	Material mat = objects[c.object_index].material;

	vec4 ambient = vec4(0.0);
	vec4 diffuse = vec4(0.0);
	vec4 specular = vec4(0.0);

	// Iterating through all lights in the scene
	for(int j = 0; j < lights_count; j++)
	{
		// Adding the light's ambient contribution
		ambient += lights[j].ambient * mat.ambient;

		// Computing the direction from the surface to the light
		vec3 light_dir = normalize(lights[j].position - c.p);

		// Check to see if any object is casting a shadow on this surface
		Ray light_ray;
		light_ray.start = c.p + c.n * 0.01;
		light_ray.dir = lights[j].position - c.p;
		bool in_shadow = false;

		// Cast the ray against the scene
		Collision c_shadow = get_closest_collision(light_ray);

		// If the ray hit an object and if the hit occurred before between the surface and the light
		if(c_shadow.object_index != -1 && c_shadow.t < 1.0)
		{
			in_shadow = true;
		}

		// If this surface is in shadow, don't add diffuse and specular components
		if(in_shadow == false)
		{
			// Computing the light's reflection on the surface
			vec3 light_ref = normalize( reflect(-light_dir, c.n));
			float cos_theta = dot(light_dir, c.n);
			float cos_phi = dot( normalize(-r.dir), light_ref);

			// Adding the light's diffuse contribution
			diffuse += lights[j].diffuse * mat.diffuse * max(cos_theta, 0.0);
			// Adding the light's specular contribution
			specular += lights[j].specular * mat.specular * pow( max( cos_phi, 0.0), mat.shininess);
		}
	}

	// Adding all of the light components to produce the final ADS Phong color
	vec4 phong_color = ambient + diffuse + specular + mat.emissive;

	return phong_color.rgb * phong_color.a;
}

//==================================================================================================================

//==================================================================================================================
// Recursive Raytrace Code
//==================================================================================================================

struct Stack_Element
{
	int type;					// The type of ray ( 1 = reflected, 2 = refracted )
	int depth;					// The depth of the recursive raytrace
	int counter;				// Keeps track of what phase each recursive call is at (Each call is broken down into four phases)

	vec3 phong_color;			// Contains the Phong ADS model color
	vec3 reflected_color;		// Contains the reflected color
	vec3 refracted_color;		// Contains the refracted color

	bool reflected;				// Whether or not this raytrace cast a reflection ray
	bool refracted;				// Whether or not this raytrace cast a refraction ray

	vec3 final_color;			// Contains the final mixed output of the recursive raytrace call
	Ray ray;					// The ray for this raytrace invocation
	Collision collision;		// The collision for this raytrace invocation. Contains a null_collision value until the phase 0
};


const int RAY_TYPE_REFLECTION = 1;
const int RAY_TYPE_REFRACTION = 2;

// Defining the null stack element
Stack_Element null_stack_element = { 0, -1, -1, vec3(0.0), vec3(0.0), vec3(0.0), false, false, vec3(0.0), null_ray, null_collision };

Stack_Element stack[100];
const int stack_size = 100;

// Points to the next free Stack_Element pointer
int stack_pointer = 0;

// Holds the last popped element from the stack
Stack_Element popped_stack_element;


//------------------------------------------------------------------------------
// Schedules a new raytrace by adding it to the top of the stack
//------------------------------------------------------------------------------
void push_stack_element(Ray r, int depth, int type)
{
	// Don't exceed the stack limits
	if(stack_pointer >= stack_size)
	{
		return;
	}

	Stack_Element element;
	element.type = type;
	element.depth = depth;
	element.counter = 0;
	element.phong_color = vec3(0.0);
	element.reflected_color = vec3(0.0);
	element.refracted_color = vec3(0.0);
	element.reflected = false;
	element.refracted = false;
	element.final_color = vec3(0.0);
	element.ray = r;
	element.collision = null_collision;

	stack[stack_pointer] = element;
	stack_pointer++;
}

//------------------------------------------------------------------------------
// Removes the topmost stack element
//------------------------------------------------------------------------------
void pop_stack_element()
{
	// Decrement the stack pointer
	stack_pointer--;
	// Store the element we're removing in popped_stack_element
	popped_stack_element = stack[stack_pointer];
	// Erase the element from the stack
	stack[stack_pointer] = null_stack_element;
}



//------------------------------------------------------------------------------
// This function processes the stack element at a given index
// This function is guaranteed to be ran on the topmost stack element
//------------------------------------------------------------------------------
void process_stack_element(int index)
{
	// If there is a popped_stack_element that just ran, it holds one of our values
	// Store it and delete it
	if(popped_stack_element != null_stack_element)
	{
		if(popped_stack_element.type == RAY_TYPE_REFLECTION)
		{
			stack[index].reflected_color = popped_stack_element.final_color;
		}
		else if(popped_stack_element.type == RAY_TYPE_REFRACTION)
		{
			stack[index].refracted_color = popped_stack_element.final_color;
		}
		popped_stack_element = null_stack_element;
	}

	Ray r = stack[index].ray;
	Collision c = stack[index].collision;

	// Iterate through the raytrace phases (explained below)
	while(stack[index].counter < 5)
	{
		//=================================================
		// PHASE 0 - Raytrace Collision Detection
		//=================================================
		if(stack[index].counter == 0)
		{
			//Cast ray against the scene, store the collision result
			c = get_closest_collision(r);

			// If the ray didn't hit anything, stop.
			if(c.object_index == -1)
				break;

			// Store the collision result
			stack[index].collision = c;
		}

		//=================================================
		// PHASE 1 - Phong ADS Lighting Computation
		//=================================================
		if(stack[index].counter == 1)
		{
			stack[index].phong_color = ads_phong_lighting(r, c);
		}
		//=================================================
		// PHASE 2 - Reflection Bounce Pass Computation
		//=================================================
		if(stack[index].counter == 2)
		{
			// Only make recursive raytrace passes if we're not at max depth
			if(stack[index].depth > 0)
			{
				// Only cast a reflection ray if the object is reflective
				if(objects[c.object_index].material.reflectivity > 0.0)
				{
					// Building the reflected ray
					Ray reflected_ray;
					reflected_ray.start = c.p + c.n * 0.001;
					reflected_ray.dir = reflect(r.dir, c.n);

					// Adding a raytrace for that ray to the stack
					push_stack_element(reflected_ray, stack[index].depth - 1, RAY_TYPE_REFLECTION);
					stack[index].reflected = true;
				}
			}
		}
		//=================================================
		// PHASE 3 - Refraction Transparency Pass Computation
		//=================================================
		if(stack[index].counter == 3)
		{
			// Only make recursive raytrace passes if we're not at max depth
			if(stack[index].depth > 0)
			{
				// Only cast a refraction ray if the object is transparent
				if(objects[c.object_index].material.transparency > 0.0)
				{
					// Building the refracted ray
					Ray refracted_ray;
					refracted_ray.start = c.p - c.n * 0.001;

					// To compute the refracted_ray direction, we need to refract the ray using the object's index of refraction
					// Calculating the ratio of indices of refraction
					float object_index_of_refraction = objects[c.object_index].material.refraction_index;
					float refraction_ratio = air_index_of_refraction / object_index_of_refraction;

					// If the ray is leaving an object from inside, flip the ratio
					if(c.inside)
					{
						refraction_ratio = 1.0 / refraction_ratio;
					}
					refracted_ray.dir = refract(r.dir, c.n, refraction_ratio);

					// Adding a raytrace for that ray to the stack
					push_stack_element(refracted_ray, stack[index].depth - 1, RAY_TYPE_REFRACTION);
					stack[index].refracted = true;
				}
			}
		}
		//=================================================
		// PHASE 4 - Mixing to produce the final color
		//=================================================
		if(stack[index].counter == 4)
		{
			Material mat = objects[c.object_index].material;

			vec3 phong_color = stack[index].phong_color;
			vec3 reflected_color = stack[index].reflected_color;
			vec3 refracted_color = stack[index].refracted_color;

			vec3 final_color = phong_color;

			if(stack[index].reflected)
			{
				final_color = mix(final_color, reflected_color, mat.reflectivity);
			}
			if(stack[index].refracted)
			{
				final_color = mix(final_color, refracted_color, mat.transparency);
			}

			stack[index].final_color = final_color;
		}
		//=================================================

		stack[index].counter++;

		// Only process one phase per process_stack_element() invocation
		return;
	}

	// Once we've finished processing the stack element, pop it
	pop_stack_element();
}


//------------------------------------------------------------------------------
// This function emulates recursive calls to raytrace for any desired depth
//------------------------------------------------------------------------------
vec3 recursive_raytrace(Ray r, int max_depth)
{
	// Add a raytrace to the stack
	push_stack_element( r, max_depth, RAY_TYPE_REFLECTION);

	// Try and catch runaway loops
	int break_counter = 0;
	int break_at = 10000;

	// Process the stack until it's empty
	while(stack_pointer > 0)
	{
		// Try and catch runaway loops
		if(break_counter++ > break_at)
		{
			break;
		}

		// Peek at the topmost stack element
		int element_index = stack_pointer - 1;

		// Process this stack element
		process_stack_element(element_index);
	}

	// If this loop was terminated because of a runaway loop
	// Return a red error color
	if(break_counter > break_at)
	{
		return vec3(1.0,0.0,0.0);
	}

	// Return the final_color value of the last-popped stack element
	return popped_stack_element.final_color;
}
//==================================================================================================================
