shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

// Surface shading for a planet's terrain mesh.
//
// The mesh carries a hypsometric colour per vertex and nothing else -- no UVs,
// no textures -- which reads well from orbit and turns into a featureless
// painted dome as soon as you fly close to it. Everything here exists to give
// that dome something to look at from a hundred metres up:
//
//   * triplanar detail noise, because the mesh has no UVs to map anything with
//     and a sphere cannot be unwrapped without a seam;
//   * a slope term, so cliffs are bare rock and only the flats keep their
//     biome colour -- the strongest single cue that a surface has relief;
//   * a macro tint, so one continent is not one flat colour.
//
// The noise is sampled in PLANET-LOCAL space rather than world space, so it
// stays pinned to the ground while the planet spins underneath it.

uniform sampler2D detail_noise : hint_default_white, filter_linear_mipmap, repeat_enable;
uniform sampler2D macro_noise : hint_default_white, filter_linear_mipmap, repeat_enable;

// Cycles per world unit. Small numbers are big features.
uniform float detail_scale = 0.35;
uniform float macro_scale = 0.004;
uniform float detail_strength : hint_range(0.0, 1.0) = 0.22;
uniform float macro_strength : hint_range(0.0, 1.0) = 0.16;

// How far a face has to tip off level before it reads as rock rather than
// ground. Level on a sphere means facing along the local up, which is just the
// vertex position, so this needs no extra per-vertex data.
uniform float rock_slope : hint_range(0.05, 1.0) = 0.42;
uniform vec3 rock_color : source_color = vec3(0.33, 0.30, 0.27);
uniform float rock_blend : hint_range(0.0, 1.0) = 0.85;

// Bumping the shading normal by the noise gradient: cheap, and the difference
// between ground that catches the light and ground that does not.
uniform float bump_strength : hint_range(0.0, 4.0) = 1.1;

// Ground that emits its own light below a given radius -- lava in the valleys
// of a volcanic world. Off by default, and off by having zero strength rather
// than a branch, because every other planet still pays for the uniform either
// way and a branch would cost more than the multiply.
uniform vec3 emission_color : source_color = vec3(1.0, 0.35, 0.05);
uniform float emission_radius = 0.0;
uniform float emission_depth = 1.0;
uniform float emission_strength : hint_range(0.0, 8.0) = 0.0;

varying vec3 local_position;
varying vec3 local_normal;

void vertex() {
	local_position = VERTEX;
	local_normal = NORMAL;
}

// Blend three planar samples by how much the surface faces each axis. The
// weights are raised to a power first: linear weights leave a visible smear
// along the diagonals, where all three contribute equally.
float triplanar(sampler2D tex, vec3 p, vec3 n) {
	vec3 w = pow(abs(n), vec3(4.0));
	w /= (w.x + w.y + w.z);
	return texture(tex, p.yz).r * w.x
		+ texture(tex, p.zx).r * w.y
		+ texture(tex, p.xy).r * w.z;
}

void fragment() {
	vec3 up = normalize(local_position);
	vec3 n = normalize(local_normal);

	float detail = triplanar(detail_noise, local_position * detail_scale, up);
	float macro = triplanar(macro_noise, local_position * macro_scale, up);

	// On a sphere the outward radial is level, so anything else is a slope.
	float slope = 1.0 - clamp(dot(n, up), 0.0, 1.0);
	float rock = smoothstep(rock_slope * 0.45, rock_slope, slope) * rock_blend;

	// Both modulations are biased to darken. A symmetric one centred on 1.0
	// brightens as often as it darkens, and two of them stacked take a pale
	// albedo -- a coastal plain, a desert -- straight past 1.0 into clipped
	// white, which is how the temperate world came out looking bleached.
	vec3 base = COLOR.rgb;
	base *= 1.0 - macro_strength + macro * macro_strength * 1.4;
	base *= 1.0 - detail_strength + detail * detail_strength * 1.4;
	base = mix(base, rock_color * (0.75 + 0.5 * detail), rock);

	ALBEDO = base;
	ROUGHNESS = mix(0.95, 0.8, rock);
	SPECULAR = 0.15;

	// Deeper than emission_radius, the ground glows. Modulated by the detail
	// noise so it breaks into veins and pools instead of washing the whole
	// basin one flat orange.
	float depth = clamp((emission_radius - length(local_position))
		/ max(emission_depth, 0.001), 0.0, 1.0);
	EMISSION = emission_color * emission_strength * depth * depth
		* smoothstep(0.35, 0.75, detail);

	// Perturb the shading normal along the screen-space gradient of the detail
	// noise. Derivatives are free here -- the GPU is already computing them to
	// pick a mip level above -- and they fade out on their own with distance,
	// which is exactly what a detail bump should do.
	vec3 tangent = normalize(cross(NORMAL, vec3(0.0, 0.0, 1.0)) + vec3(1e-5));
	vec3 bitangent = normalize(cross(NORMAL, tangent));
	NORMAL = normalize(NORMAL - bump_strength * (
		dFdx(detail) * tangent + dFdy(detail) * bitangent));
}
