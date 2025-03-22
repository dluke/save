shader_type canvas_item;

uniform sampler2D spritesheet;
uniform sampler2D random_noise : hint_default_black;

uniform sampler2D grass_color;

uniform int frames = 4;

varying vec2 world_pos;
void vertex() {
	world_pos = INSTANCE_CUSTOM.xy;
}

void fragment() {
	// Get random number using world position
	vec2 rand_uv = world_pos;
	vec2 rand_tex_size = vec2(textureSize(random_noise, 0));
	rand_uv.x = mod(rand_uv.x, rand_tex_size.x);
	rand_uv.y = mod(rand_uv.y, rand_tex_size.y);
	float rand = texelFetch(random_noise, ivec2(rand_uv), 0).r;
	float ff = float(frames);
	rand = mod(round(rand * ff), 4);
	
	// Pick a random grass blade type from spritesheet
	float frame_number = rand;
	float fr = 1.0 / float(max(frames, 1));
	float x_coord = mix(fr * frame_number, fr * frame_number + fr, UV.x);
	vec2 uv = vec2(x_coord, UV.y);
	
	// Coloring
	vec4 tex_col = texture(spritesheet, uv);
	vec4 bottom_color = texelFetch(grass_color, ivec2(0,0), 0);
	vec4 top_color = texelFetch(grass_color, ivec2(255,0), 0);
	vec3 color = mix(top_color.rgb, bottom_color.rgb, UV.y);
	color *= tex_col.rgb;
	
	COLOR = vec4(color, tex_col.a);
//	COLOR = vec4(color, tex_col.a * player_mask);
}

