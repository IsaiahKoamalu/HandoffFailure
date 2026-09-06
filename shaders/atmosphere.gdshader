shader_type spatial;
render_mode unshaded, blend_add, cull_back, depth_draw_never, shadows_disabled;

// A shell a little larger than the planet, drawn as pure added light.
//
// Two terms, and both are load-bearing. The rim term stands in for how much air
// a view ray passes through: largest at the limb, near zero looking straight
// down, which is what makes the edge of a planet glow. The sun term is whether
// that air is lit at all -- without it a planet's night side wears a bright
// halo and the whole thing reads as a lamp rather than a world.
//
// The star sits at the origin of the system and never moves, so its position is
// a constant written once at build time rather than a uniform pushed every
// frame.
//
// cull_back, not cull_disabled. With both faces drawn, the near face is in
// front of the terrain and adds over the disc while the far face is behind it
// and is depth-rejected -- so the halo doubles in exactly the ring outside the
// planet's silhouette and nowhere else, and the seam between one layer and two
// draws a hard circle around the world. One face has no seam to draw.

uniform vec3 atmosphere_color : source_color = vec3(0.35, 0.55, 1.0);
uniform vec3 sunset_color : source_color = vec3(1.0, 0.55, 0.28);
uniform vec3 sun_position = vec3(0.0, 0.0, 0.0);
uniform float density : hint_range(0.0, 4.0) = 1.0;
// Higher is a tighter band of glow hugging the limb.
uniform float falloff : hint_range(0.5, 12.0) = 2.4;
// A floor under the rim term, so the lit face of the world keeps a thin veil of
// air over it instead of the atmosphere being a ring around an airless planet.
uniform float haze : hint_range(0.0, 1.0) = 0.045;

void fragment() {
	vec3 world_position = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 world_normal = normalize((INV_VIEW_MATRIX * vec4(NORMAL, 0.0)).xyz);
	vec3 to_camera = normalize(INV_VIEW_MATRIX[3].xyz - world_position);
	vec3 to_sun = normalize(sun_position - world_position);

	float rim = 1.0 - abs(dot(world_normal, to_camera));
	float thickness = pow(clamp(rim, 0.0, 1.0), falloff) + haze;

	// Wrapped rather than clamped at zero: the glow has to survive a little way
	// past the terminator, or the day-night edge is a hard line across the halo.
	float lit = clamp(dot(world_normal, to_sun) * 0.6 + 0.4, 0.0, 1.0);

	// Grazing light reddens, which is the whole of what a sunrise is.
	float grazing = 1.0 - abs(dot(to_sun, to_camera));
	vec3 tint = mix(atmosphere_color, sunset_color,
		clamp(grazing * (1.0 - lit) * 2.0, 0.0, 1.0));

	ALBEDO = tint;
	ALPHA = thickness * lit * density;
}
