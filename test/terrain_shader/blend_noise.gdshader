shader_type canvas_item;

uniform sampler2D noise_texture;
uniform float threshold;

const vec4 WHITE = vec4(1.0, 1.0, 1.0, 1.0);
const vec4 RED = vec4(1.0, 0.0, 0.0, 1.0);

void fragment() {
	vec2 texture_size = vec2(textureSize(noise_texture, 0));
	ivec2 pixel_coords = ivec2(UV * texture_size);
	vec4 noise = texelFetch(noise_texture, pixel_coords, 0);
	vec4 mask = texture(TEXTURE, UV);
	vec4 value = mask.a > threshold ? WHITE : mask * noise;
//	vec4 value = mask * noise;
//	vec4 value = mask.a > threshold ? WHITE : mask;
	COLOR = value;
}
