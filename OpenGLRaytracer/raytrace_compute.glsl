#version 430

// Time (seconds)
uniform float time;

// The output texture
uniform image2D output_texture;

// Each group runs on one pixel
layout (local_size_x = 1, local_size_y = 1) in;

const float PI = 3.14159265358;
const float HALF_PI = PI * 0.5;
const float TWO_PI = PI * 2;
const float ROOT_TWO_PI = sqrt(TWO_PI);
const float DEG_TO_RAD = PI / 180.0;
const float RAD_TO_DEG = 180.0 / PI;


// How many importance-sampling bounce passes to do per ray-surface collision
const int IMPORTANCE_SAMPLING_SAMPLES = 60;


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
vec3 raytrace(Ray r, int depth);
float inv_cdf(float p);
float gaussian_brdf(float theta_i, float theta_o, float r);
float rand(vec2 st);
float rand_in_range( vec2 st, float min_v, float max_v);


// Defining the null object types
Box null_box = { vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0) };

// How much to scale the box movements by
float time_scale = 0.4;


float scaled_time = time * time_scale;


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
			vec3(-1.0,-1.0,-1.0) * (0.5*sin(scaled_time * 0.5) + 1.5),
			vec3( 1.0, 1.0, 1.0) * (0.5*sin(scaled_time * 0.5) + 1.5)
		},
		// Make the box bob up and down
		vec3( 0.0, 0.0, sin(scaled_time * 3.0)),
		vec3( 0.0, scaled_time * 90.0, 0.0),
		vec3( 1.0, 0.0, 0.0)
	},
	{
		{
			vec3(-10.0,-10.0,-1.0),
			vec3( 10.0, 10.0, 1.0)
		},
		vec3( 0.0, 0.0,-3.0),
		// Make the box lean from one side to the other
		vec3( sin(scaled_time * 5.0) * 10.0, 45.0, 0.0),
		vec3( 0.0, 1.0, 0.0)
	},
	{
		{
			vec3(-1.0,-1.0,-2.0),
			vec3( 1.0, 1.0, 2.0)
		},
		vec3( 3.0, 4.0, 4.0),
		// Make the box tumble in the air
		vec3( 45.0 + scaled_time * 45.0, 0.0, 45.0 + scaled_time * 180.0),
		vec3( 0.0, 0.0, 1.0)
	},
};

int objects_count = 4;

vec3 recursive_raytrace();

void main()
{
	int width = int(gl_NumWorkGroups.x);
	int height = int(gl_NumWorkGroups.y);
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

	if(pixel.x >= 512 || pixel.y >= 288)
		return;

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

	vec3 color = raytrace(world_ray,1);

	//color = recursive_raytrace();

	//========================================
	// RNG Test Function
	//========================================

	//========================================

	vec3 final_color = color;

	imageStore(output_texture, pixel, vec4(final_color,0.0));
}

// Computes and returns a perspective projection matrix
// c : the camera for which to compute the projection matrix
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
	float cos_deg = cos(DEG_TO_RAD * deg);
	float sin_deg = sin(DEG_TO_RAD * deg);
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
	float cos_deg = cos(DEG_TO_RAD * deg);
	float sin_deg = sin(DEG_TO_RAD * deg);
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
	float cos_deg = cos(DEG_TO_RAD * deg);
	float sin_deg = sin(DEG_TO_RAD * deg);
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

// Computes a rotation matrix given a 3D rotation
// angle radians about an arbitrary axis
// angle : the rotation in radians
// axis : the axis of rotation
// Implementation found on:
// http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
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

struct Collision
{
	float t;
	vec3 p;
	vec3 n;
};

// This defines what a null collision looks like
Collision null_collision = { -1.0, vec3(0.0), vec3(0.0)};


//FIXME - this is a duplicate function to alleviate the no recursion limitation
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

	// intersects if t_near > 0 && t_near < t_far
	//return vec2(t_near, t_far);

	Collision c;

	c.t = t_near;

	// If the ray didn't intersect the box:
	if(t_near >= t_far)
	{
		c.t = -1.0;
		return c;
	}

	// Getting the face normal of the intersection:
	int face_index = 0;
	if(t_near == t1.y)
		face_index = 1;
	else if(t_near == t1.z)
		face_index = 2;
	c.n = vec3(0.0);
	c.n[face_index] = 1.0;
	// If we hit the box from the negative axis, invert the normal
	if(ray_start[face_index] < 0.0)
		c.n *= -1.0;
	// Converting the normal to world-space
	c.n = transpose(inverse(mat3(local_to_world))) * c.n;

	// Calculate the world-position of the intersection:
	c.p = (local_to_world * vec4(ray_start + t_near * ray_dir,1.0)).xyz;
	// Recalculating c.t in world-space
	//FIXME - this doesn't quite work
	//c.t = length(c.p - r.start);


	//TODO - We need to catch rays leaving the object too, 
	// I beleive t_far may have the exit info,
	// we if(t_near < 0.0), then  we run the face detection code on t_far and return that, inverting the normal (maybe)

	return c;
}

vec3 raytrace2(Ray r, int depth)
{
	if(depth < 0)
		return vec3(0.0);

	float closest = 10000;
	int closest_index = -1;
	Collision closest_collision;

	for(int i = 0; i < objects_count; i++)
	{
		Collision c;
		
		// If the object is a box
		if(objects[i].box != null_box)
		{
			//vec2 lambda = intersect_box_object(r, objects[i]);
			// If the ray intersects the box
			//if(lambda.x > 0.0 && lambda.x < lambda.y)
			c = intersect_box_object(r, objects[i]);
			if(c.t <= 0.0)
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

		if(c.t < closest)
		{
			closest = c.t;
			closest_index = i;
			closest_collision = c;
		}
	}

	if(closest_index != -1)
	{
		//return objects[closest_index].color;
		//return closest_collision.n * 0.5 + vec3(0.5);
		//return vec3(closest_collision.t / 100.0);
		//return closest_collision.p / 10.0;
		Ray bounce;
		bounce.start = closest_collision.p;
		bounce.dir = reflect(r.dir, closest_collision.n);
	
		//NOOOO - recursion is not allowed in GLSL....
		return objects[closest_index].color;
	}
	return vec3(0.0,0.0,0.0);
}

vec3 raytrace(Ray r, int depth)
{
	if(depth < 0)
		return vec3(0.0);

	float closest = 10000;
	int closest_index = -1;
	Collision closest_collision;

	for(int i = 0; i < objects_count; i++)
	{
		Collision c;
		
		// If the object is a box
		if(objects[i].box != null_box)
		{
			//vec2 lambda = intersect_box_object(r, objects[i]);
			// If the ray intersects the box
			//if(lambda.x > 0.0 && lambda.x < lambda.y)
			c = intersect_box_object(r, objects[i]);
			if(c.t <= 0.0)
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

		if(c.t < closest)
		{
			closest = c.t;
			closest_index = i;
			closest_collision = c;
		}
	}

	if(closest_index != -1)
	{
		//return objects[closest_index].color;
		//return closest_collision.n * 0.5 + vec3(0.5);
		//return vec3(closest_collision.t / 100.0);
		//return closest_collision.p / 10.0;
		//Ray bounce;
		//bounce.start = closest_collision.p;
		//bounce.dir = reflect(r.dir, closest_collision.n);
	
		//NOOOO - recursion is not allowed in GLSL....
		// TODO - implement BRDF, importance-sample some number of rays about the reflected angle.


		// what are the important rays?
		// Should I do it at percentiles? (20,40,60,80,100)
		// Should I just sample ever 30 degrees and then average the contribution?
		//----------------------------------------------------------------------------
		//				Sample IMPORTANCE_SAMPLING_SAMPLES times, 
		//				use BRDF to modulate each ray's contribution
		//----------------------------------------------------------------------------
		// The angle between the incident ray and the surface normal
		float theta_i = acos(dot( -r.dir, closest_collision.n));
		

		//float surface_roughness = mod(time * 0.1, 1.0);
		float surface_roughness = sin(time * 0.5) * 0.5 + 0.501;
		surface_roughness = 1.0;
		//surface_roughness = 0.001;
		//surface_roughness = 0.3;
		vec3 color = vec3(0.0);

		// How much intensity each pass contributed
		// Use to scale the output so that the the total intensity does not exceed 1.0
		float total_intensity = 0.0;

		
		/*float percentile_width = 1.0 / (float(IMPORTANCE_SAMPLING_SAMPLES) + 1.0);
		//------------ Dithering Precomputations ------------
//		float phi_dither = percentile_width * PI;
//		vec2 phi_prng_seed = r.start.xz * r.dir.xz;
//		float phi_dither_percent = rand(phi_prng_seed);
//		phi_dither *= (phi_dither_percent > 0.5 ? 1.0 : -1.0);
//
//		float theta_dither = percentile_width * HALF_PI;
//		vec2 theta_prng_seed = r.start.xy * r.dir.xy;
//		float theta_dither_percent = rand(theta_prng_seed);
//		theta_dither *= (theta_dither_percent > 0.5 ? 1.0 : -1.0);
		//---------------------------------------------------

		const float dither_strength = sin(time * 0.5) * 0.5 + 0.501;;

		for(int j = 0; j < IMPORTANCE_SAMPLING_SAMPLES; j++)
		{
				float phi_percentile = percentile_width	* float(j + 1);
				

				for(int k = 0; k < IMPORTANCE_SAMPLING_SAMPLES; k++)
				{
					// All of this can be moved above:
					//====================================
					// Dithering
					float phi_dither = percentile_width * PI;
					vec2 phi_prng_seed = r.start.xz * r.dir.xz * ((j+1)*(k+1) / IMPORTANCE_SAMPLING_SAMPLES);
					float phi_dither_percent = rand(phi_prng_seed);
					phi_dither *= dither_strength*(phi_dither_percent > 0.5 ? 1.0 : -1.0);

					phi_percentile += phi_dither;

					// Calculate the z-score for this percentile
					float phi_z_score = inv_cdf(phi_percentile);

					// Calculate phi (the angle that the outbound ray is rotated ABOUT the normal)
					// centered at 0, with standard deviation at pi * surface_roughness
					float phi_o = phi_z_score * PI * surface_roughness;
					//====================================


					float theta_percentile = percentile_width * float(j + 1);
					// Dithering
					float theta_dither = percentile_width * HALF_PI;
					vec2 theta_prng_seed = r.start.xy * r.dir.xy * ((j+1)*(k+1) / IMPORTANCE_SAMPLING_SAMPLES);
					float theta_dither_percent = rand(theta_prng_seed);
					theta_dither *= dither_strength*(theta_dither_percent > 0.5 ? 1.0 : -1.0);
					
					theta_percentile += theta_dither;



					// Calculate the z-score for this percentile
					float theta_z_score = inv_cdf(theta_percentile);


					// Convert the z-score to the BRDF Gaussian Distribution
					// Make the contribution of the incident angle falloff as the surface 
					// roughness approaches 1
					// Calculate theta (the angle the outbound ray makes with the normal)
					float theta_o = theta_z_score * surface_roughness + (theta_i * (1 - surface_roughness));


					//-------------- Dithering ----------------
					//theta_o += theta_dither;
					//phi_o += phi_dither;
					//-----------------------------------------
					
					// Compute the intensity of this ray
					float intensity = gaussian_brdf(theta_i, theta_o, surface_roughness);
					total_intensity += intensity;
				
					// Calculate the reflected ray
					Ray bounce;
					bounce.start = closest_collision.p;
					vec3 axis = normalize(cross( -r.dir, closest_collision.n));
					bounce.dir = rotation_matrix(closest_collision.n, phi_o) * rotation_matrix(axis, theta_o) * closest_collision.n;


					color += raytrace2( bounce, depth - 1) * intensity;
					//======================================
					// TEST : comparing this code to the single-bounce pass code
					// The red and green components should match
					//======================================
//					vec3 trace_color = raytrace2( bounce, depth - 1);
					// Getting the calced raytrace:
//					color.x += max(max(trace_color.x, trace_color.y), trace_color.z);
//					// Calculating the actual bounce
//					Ray baseline_bounce = {closest_collision.p, reflect(r.dir, closest_collision.n) };
//					vec3 baseline_col = raytrace2( baseline_bounce, depth - 1);
//					float baseline_val = max(max(baseline_col.x,baseline_col.y),baseline_col.z);
//					color.y += baseline_val;

					//return bounce.dir * 0.5 + vec3(0.5);
				}
		}*/

		// Random importance sampling:
		int additional_rays = 0;
		for(int j = 0; j < IMPORTANCE_SAMPLING_SAMPLES; j++)
		{
				//------------ Calculating Phi -------------
				// Coming up with a seed that's unique per ray, per pass
				vec2 phi_prng_seed = r.start.xz * r.dir.xz * (j + 1 + additional_rays) / IMPORTANCE_SAMPLING_SAMPLES;
				// Choosing a random percentile to use for this ray's phi angle
				float phi_percentile = rand_in_range(phi_prng_seed, 0.01, 0.99);

				// Calculate the z-score for this percentile
				float phi_z_score = inv_cdf(phi_percentile);

				// Calculate phi (the angle that the outbound ray is rotated ABOUT the normal)
				// centered at 0, with standard deviation at pi * surface_roughness
				float phi_o = phi_z_score * PI * surface_roughness;
				//------------------------------------------

				//------------ Calculating Theta -------------
				// Coming up with a seed that's unique per ray, per pass
				vec2 theta_prng_seed = r.start.xy * r.dir.xy * (j + 1 + additional_rays) / IMPORTANCE_SAMPLING_SAMPLES;
				
				// Choosing a random percentile to use for this ray's theta angle
				float theta_percentile = rand_in_range(theta_prng_seed, 0.01, 0.99);

				// Calculate the z-score for this percentile
				float theta_z_score = inv_cdf(theta_percentile);

				// Convert the z-score to the BRDF Gaussian Distribution
				// Make the contribution of the incident angle falloff as the surface roughness approaches 1
				// Calculate theta (the angle the outbound ray makes with the normal)
				float theta_o = theta_z_score * surface_roughness + (theta_i * (1 - surface_roughness));
				//------------------------------------------

				// Test - Seeing what RNG values we're getting
				//return vec3(phi_percentile,theta_percentile,0.0);
				//return vec3(phi_percentile);

				// Test not using BRDF-based importance sampling
				//phi_o = (phi_percentile * 2.0 - 1.0) * PI * surface_roughness;
				//theta_o = (theta_percentile * 2.0 - 1.0) * surface_roughness + (theta_i * (1 - surface_roughness));
					
				// Compute the intensity of this ray
				float intensity = gaussian_brdf(theta_i, theta_o, surface_roughness);
				total_intensity += intensity;

				
				// Calculate the reflected ray
				Ray bounce;
				bounce.start = closest_collision.p;
				vec3 axis = normalize(cross( -r.dir, closest_collision.n));
				bounce.dir = rotation_matrix(closest_collision.n, phi_o) * rotation_matrix(axis, theta_o) * closest_collision.n;

				color += raytrace2( bounce, depth - 1) * intensity;

				// If the pixel's color value is low, 
				// allow the ray the chance to cast additional rays
				if( rand(r.start.xy * r.dir.xz * (additional_rays + 1) ) - 0.02 > dot(color,color)/total_intensity )
				{
					additional_rays++;
					j--;
				}
		}
		return color / total_intensity + vec3(0.05,0.05,0.05);
		// Dividing color by total intensity to ensure 
		color /= total_intensity;
		//return (objects[closest_index].color + color);
		//----------------------------------------------------------------------------
		//return ((objects[closest_index].color + raytrace2( bounce, depth - 1))) * 0.5;
	}
	return vec3(0.0,0.0,0.0);
}


//==================================================================================================================
// Bi-Directional Reflectance Distribution Function (BRDF)
//==================================================================================================================

//---------------------------------------------------------
// The inverse of the Cumulative Distribution Function 
// for the Gaussian Normal Distribution Function
//---------------------------------------------------------
// This implementation is based on an approximation by 
// Paul M. Voutier, as outlined in his paper: 
// https://arxiv.org/abs/1002.0567
//---------------------------------------------------------
// This function takes a desired percentile as a parameter
// and outputs a z-score corresponding to the x-coordinate 
// on the standard gaussian function graph
// p : the desired percentile whose z-score we want
//---------------------------------------------------------
float inv_cdf(float p) 
{
	// The central section
	if(0.0465 <= p && p <= 0.9535)
	{
		float a[] =
		{
			1.246899760652504, 0.195740115269792,-0.652871358365296, 
			0.155331081623168, -0.839293158122257
		};
		float q = p - 0.5;
		float r = q * q;
		return q * (a[0] + (((a[2] * r) + (a[1])) / ((r * r) + (a[4] * r) + (a[3]))));
	}
	// The left and right tail sections:
	else
	{
		float b[] =
		{
			-1.000182518730158122,16.682320830719986527,4.120411523939115059,
			0.029814187308200211,7.173787663925508066,8.759693508958633869
		};
		// A scaling factor I introduced for better results
		float scale = 1.64;
		// If we're dealing with the right-tail section, invert p
		if(p > 0.5)
		{
			p = p - 1.0;
			scale *= -1.0;
		}
		float q = sqrt(log(1.0 / (p * p)));
		return scale * (b[0] * q) + (b[3]) + (((b[2] * q) + (b[1])) / ((q * q) + (b[5] * q) + (b[4])));
	}
}
//---------------------------------------------------------

//---------------------------------------------------------
// The Gaussian Function based BRDF
//---------------------------------------------------------
// This implementation is based on the simple gaussian formula
// with standard deviation r and mean of theta_i
//---------------------------------------------------------
// This implementation is modified to have the mean tend 
// to 0 as the roughness approaches 1
//---------------------------------------------------------
// This is done to simulate the the importance of the 
// angle of the incident ray tends to decrease as the 
// surface approaches a perfect diffuser
//---------------------------------------------------------
// Takes as a parameter the surface roughness, incident 
// and outgoing (reflected) angles and outputs the 
// intensity of the ray from 0 to 1
// theta_i : the angle of the incident ray [0,pi/2]
// theta_o : the angle of the reflected ray [-pi/2, pi/2]
// r : the surface roughness (0,1]
//---------------------------------------------------------
float gaussian_brdf(float theta_i, float theta_o, float r)
{
	float inv_r = 1.0 / r;
	// Make the theta_i contribution tend to 0 as r tends to 1
	float a = theta_o - (theta_i * (1 - r));
	float exponent = -0.5 * inv_r * inv_r * a * a;
	return inv_r * (1.0 / ROOT_TWO_PI) * exp(exponent);
}
//---------------------------------------------------------

//---------------------------------------------------------
// Pseudo Random Number Generator
//---------------------------------------------------------
// This implementation can be found on
// https://thebookofshaders.com/10/
//---------------------------------------------------------
// Returns a uniformly distributed random number in [0,1)
// given seeds s and t as a vec2
//---------------------------------------------------------
float rand(vec2 st)
{
	// Suggested values:
	//const float a = 12.9898;
	//const float b = 78.233;
	//const float c = 43758.5453123;

	// Using my own custom values that seem to give a more uniform distribution:
	// NOTE: a histogram of the PRNG with these values shows a small spike near the 
	// 50th - 60th percentile, but this spike isn't significant enough to skew our results
	const float a = 51.3259;
	const float b = 75.152;
	const float c = 4087.62;

	return fract(sin(dot(st, vec2(a, b))) * c);
}
//---------------------------------------------------------

//---------------------------------------------------------
// ranged-Pseudo Random Number Generator
//---------------------------------------------------------
// Returns a uniformly distributed random number  
// in [min_v,max_v), given seeds s and t as a vec2
//---------------------------------------------------------
// Returns a random number between min_v and max_v
float rand_in_range( vec2 st, float min_v, float max_v)
{
	return min_v + (rand(st) * (max_v - min_v));
}
//---------------------------------------------------------

//==================================================================================================================

//==================================================================================================================
// Recursive Raytrace Code
//==================================================================================================================
struct Raytrace_Stack_Element
{
	// Function Call Parameters
	Ray r;
	int depth;

	// Function State Parameters
	Collision c;

	// TODO - somehow reference the line of code that added this stack element to the stack 
	// ( we need to know what portion / line of code in the calling stack element to return to)
	int return_address;
	// The index of the calling stack entry
	int return_address_index;
};

// Defining the null raytrace_stack_element
Raytrace_Stack_Element null_raytrace_stack_element = { null_ray, -1, null_collision, -1, -1};


Raytrace_Stack_Element stack[100];
int stack_capacity = 100;

// Index of the last element written into the stack
int stack_pointer = -1;





// Adds a recursive raytrace call to our stack
// returns false if the stack is full.
// return_address : the index of the calling element (is -1 for the first call)
bool schedule_recursive_raytrace(Ray r, int depth, int return_address)
{
	if(stack_pointer >= stack_capacity - 1)
		return false;

	stack_pointer++;

	stack[stack_pointer].r = r;
	stack[stack_pointer].depth = depth;
	stack[stack_pointer].return_address = return_address;
	
	return true;
}


// A test recursive function
void test_func(Ray r, int depth)
{
	if(depth >= 1)
		return;

	// Pushing a new stack_element to the stack
	schedule_recursive_raytrace(r,depth + 1, -1);
}

// This function emulates recursive calls to raytrace:
//FIXME - should be void
vec3 recursive_raytrace()
{
		//TODO - need to iterate through the stack
		// until we hit a stack overflow
		// or until the stack is empty

		// The current process we are running
		//int program_counter = 0;

		// Starting the first call:
		//stack[0].r = null_ray;
		//stack[0].c = null_collision;
		stack[0].depth = 0;
		//stack[0].return_address = -1;
		//stack[0].return_address_index = -1;

		stack_pointer = 0;

		int val = 0;

		// While the stack is not empty:
		while(stack_pointer >= 0)
		{
			// Getting the current stack_element from the stack to execute:
			Raytrace_Stack_Element stack_element = stack[stack_pointer];
			
			// Execute the stack_element
			test_func(stack_element.r, stack_element.depth);
			//TODO - if the call added a new raytrace stack, we need to continue this loop:
			if(stack_element.depth < 1)
				continue;
			
			// FIXME- but now when we get back to the first function, it calls again...
			val++;

			// Popping the current stack_element
			stack[stack_pointer] = null_raytrace_stack_element;

			// Decrementing the program_counter
			stack_pointer--;
		}

		// We want to debug how many runs are being executed:
		vec3 colors[] = {vec3(0.0), vec3(0.5,0,0), vec3(1,0,0), vec3(0,0.5,0), vec3(0,1,0), vec3(0,0,0.5), vec3(0,0,1)};

		if(val > 6)
			val = 6;

		return colors[val];
	
		// while we have a function 


		//pseudo-code
		//while(!stack is empty)
		//{
		//		pull process the element at the stack pointer
		//		if that element adds another stack entry, repeat
		//		else, 
		//			get its return value, 
		//			if no calling stack entry, 
		//				terminate
		//			return to calling stack entry and repeat
		//}
}
//==================================================================================================================