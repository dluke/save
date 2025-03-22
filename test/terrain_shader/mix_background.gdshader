shader_type canvas_item;

uniform sampler2D grass;
uniform sampler2D swamp;
uniform sampler2D mask;
uniform sampler2D noise_texture;
uniform float swamp_threshold;
uniform vec2 size = vec2(1920, 1080);

const vec4 WHITE = vec4(1.0, 1.0, 1.0, 1.0);

void fragment() {
	vec2 pixel_coords = vec2(UV * size);
	vec2 grass_size = vec2(textureSize(grass, 0));
	vec2 wrap_grass_coord = vec2(
		mod(pixel_coords.x, grass_size.x), 
		mod(pixel_coords.y, grass_size.y)
		);
		
	vec2 swamp_size = vec2(textureSize(swamp, 0));
	vec2 wrap_swamp_coord = vec2(
		mod(pixel_coords.x, swamp_size.x), 
		mod(pixel_coords.y, swamp_size.y)
		);
	
	vec4 mask_value = texelFetch(mask, ivec2(pixel_coords), 0);
	vec4 grass_color = texelFetch(grass, ivec2(wrap_grass_coord), 0);
	vec4 swamp_color = texelFetch(swamp, ivec2(wrap_swamp_coord), 0);
	
	float noise_value = texelFetch(noise_texture, ivec2(pixel_coords), 0).r;
	noise_value = clamp(noise_value, 0.35, 1.0);
	float swamp_value = mask_value.a * noise_value;
	swamp_value = swamp_value > swamp_threshold ? 1.0 : swamp_value;
	
	vec4 color = mix(grass_color, swamp_color, swamp_value);
	COLOR = color;
}
