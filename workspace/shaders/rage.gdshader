shader_type canvas_item;

uniform sampler2D noise_pattern : repeat_enable;
uniform vec2 scroll = vec2(0.0, 1.0);
uniform float scroll_speed = 0.2;
uniform sampler2D tone_mapping;


void fragment() {
	float intensity = texture(noise_pattern, UV + TIME * scroll_speed * scroll).a;
	COLOR = texture(tone_mapping, vec2(intensity));
//	COLOR = vec4(intensity);
}
