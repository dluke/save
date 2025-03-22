shader_type particles;

uniform float rows = 16;
uniform float spacing = 1.0;

void start() {
	// obtain our position based on which particle we're rendering
	vec3 pos = vec3(0.0, 0.0, 0.0);
	pos.y = float(INDEX);
	pos.x = mod(pos.y, rows);
	pos.y = (pos.y - pos.x) / rows; 
	
	// center this
	pos.x -= rows * 0.5;
	pos.y -= rows * 0.5;
	
	// and now apply our spacing
	pos *= spacing;
	
	// now center on our particle location but within our spacing
//	pos.x += EMISSION_TRANSFORM[3][0] - mod(EMISSION_TRANSFORM[3][0], spacing);
//	pos.y += EMISSION_TRANSFORM[3][1] - mod(EMISSION_TRANSFORM[3][1], spacing);
	pos.x += EMISSION_TRANSFORM[3][0];
	pos.y += EMISSION_TRANSFORM[3][1];
	
	// update our transform to place
	TRANSFORM[3][0] = pos.x;
	TRANSFORM[3][1] = pos.y;
	
	// Pass world position to instances
	CUSTOM.xy = pos.xy;
}

void process() {
	// Place process code here.
}
