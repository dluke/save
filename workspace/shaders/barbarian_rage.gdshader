shader_type canvas_item;

uniform sampler2D mask;
uniform sampler2D noise_pattern : repeat_enable;
uniform vec2 scroll = vec2(0.0, 1.0);
uniform float scroll_speed = 0.1;
uniform sampler2D tone_mapping;

void fragment() {
	float alpha = texture(mask, UV).a;
	float intensity = texture(noise_pattern, UV + TIME * scroll_speed * scroll).a;
	vec4 color = texture(tone_mapping, vec2(intensity));
	color.a = alpha;
	COLOR = color;
}
