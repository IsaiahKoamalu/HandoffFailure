shader_type spatial;
render_mode blend_mix, cull_disabled, depth_draw_never, diffuse_lambert;

// A thin shell of cloud above the terrain, cut out of triplanar noise.
//
// Cut out rather than painted: coverage thresholds the noise, so there is clear
// sky between the systems instead of a uniform haze, and that threshold is the
// one knob that turns a clear world into an overcast one.

uniform sampler2D cloud_noise : hint_default_white, filter_linear_mipmap, repeat_enable;
uniform vec3 cloud_color : source_color = vec3(1.0, 1.0, 1.0);
uniform float noise_scale = 0.015;
uniform float coverage : hint_range(0.0, 1.0) = 0.5;
uniform float softness : hint_range(0.01, 0.5) = 0.18;
uniform float opacity : hint_range(0.0, 1.0) = 0.85;

varying vec3 local_position;
varying vec3 local_normal;

void vertex() {
	local_position = VERTEX;
	local_normal = NORMAL;
}

float triplanar(sampler2D tex, vec3 p, vec3 n) {
	vec3 w = pow(abs(n), vec3(4.0));
	w /= (w.x + w.y + w.z);
	return texture(tex, p.yz).r * w.x
		+ texture(tex, p.zx).r * w.y
		+ texture(tex, p.xy).r * w.z;
}

void fragment() {
	vec3 up = normalize(local_position);
	float n = triplanar(cloud_noise, local_position * noise_scale, up);

	// coverage 0 leaves nothing at all, coverage 1 leaves solid overcast.
	float threshold = 1.0 - coverage;
	float mask = smoothstep(threshold, threshold + softness, n);
	// Discarding rather than drawing near-zero alpha: the shell is a full
	// sphere and the clear parts of it would otherwise still pay for blending.
	if (mask < 0.004) {
		discard;
	}
	// Fade out where the shell is edge-on to the camera. Without this the limb
	// -- where a view ray runs the long way through the shell and every sample
	// saturates -- draws a hard ring around the planet just outside the
	// terrain's silhouette, which reads as a band of dirt rather than as cloud.
	// Culling the back faces instead would fix the ring and take the sky away
	// from anyone standing underneath it.
	vec3 world_position = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 world_normal = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
	float facing = abs(dot(world_normal,
		normalize(INV_VIEW_MATRIX[3].xyz - world_position)));

	ALBEDO = cloud_color;
	ALPHA = mask * opacity * smoothstep(0.0, 0.35, facing);
	ROUGHNESS = 1.0;
}
