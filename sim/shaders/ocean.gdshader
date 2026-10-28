shader_type spatial;
render_mode blend_mix, cull_disabled, diffuse_burley, specular_schlick_ggx;

// The ocean is a smooth sphere at sea level. What sells it as water is not its
// shape but two angle-dependent terms: it goes mirror-like at grazing angles
// and keeps its own colour looking straight down, and it takes a hard specular
// highlight from the star. A flat translucent sphere has neither, and reads as
// coloured glass.
//
// cull_disabled because the surface has to survive the camera dropping below
// it -- a ship ditches in the sea and the pilot is then underneath it.

uniform vec4 shallow_color : source_color = vec4(0.10, 0.36, 0.52, 0.72);
uniform vec4 deep_color : source_color = vec4(0.02, 0.10, 0.22, 0.94);
uniform float fresnel_power : hint_range(0.5, 8.0) = 3.0;

void fragment() {
	vec3 world_position = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 world_normal = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
	vec3 to_camera = normalize(INV_VIEW_MATRIX[3].xyz - world_position);

	float facing = abs(dot(world_normal, to_camera));
	float fresnel = pow(1.0 - facing, fresnel_power);

	// Looking straight down into water shows its depth colour; looking along it
	// shows the sky. Alpha follows the same curve, so a shoreline seen from
	// above is translucent and the same water at the horizon is not.
	vec4 water = mix(deep_color, shallow_color, facing);
	ALBEDO = water.rgb;
	ALPHA = mix(water.a, 1.0, fresnel * 0.85);
	ROUGHNESS = mix(0.02, 0.25, facing);
	METALLIC = 0.0;
	SPECULAR = mix(1.0, 0.5, facing);
}
