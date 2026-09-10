shader_type spatial;
render_mode unshaded, blend_add, cull_front, depth_draw_never, shadows_disabled;

// A shell a little larger than the planet, drawn as pure added light.
//
// The brightness at a pixel is how much air the view ray passes through, worked
// out analytically: intersect the ray with the shell, subtract whatever the
// planet's own body blocks, and put the remaining path length through
// Beer-Lambert. That is one formula, and it does three jobs that a heuristic
// needed three of:
//
//   * The halo fades to nothing at the outer edge, because the chord through
//     the shell goes to zero there. The rim term this replaced was at its
//     *maximum* on that silhouette, so every planet wore a hard-edged bubble
//     with a visible gap between the terrain and the shell -- a glass marble
//     rather than a world.
//   * The limb glows, because a ray grazing the surface travels a long way
//     through air.
//   * Standing on the ground, the sky is lit, because looking up is also a path
//     through air. It costs nothing extra and it is the whole difference
//     between a planet with weather and an asteroid.
//
// cull_front, so the *inside* of the shell is what gets shaded. From outside
// that is the far hemisphere, which the planet's own depth rejects over the
// disc and leaves as a ring of haze around the silhouette. From the surface it
// is the sky overhead. Drawing the near faces instead -- which is what this did
// -- means the shell vanishes the moment you fly into it, so an atmosphere was
// something you could only see from space.
//
// Depth testing stays on, deliberately. Disabling it would let the haze paint
// over the ship, the station and anything else between the camera and the air.

uniform vec3 atmosphere_color : source_color = vec3(0.35, 0.55, 1.0);
uniform vec3 sunset_color : source_color = vec3(1.0, 0.55, 0.28);
uniform vec3 sun_position = vec3(0.0, 0.0, 0.0);
// Optical depth per scale height. At 1.0, looking straight up from the ground
// clears about 63% of the way to full colour.
uniform float density : hint_range(0.0, 4.0) = 1.0;
// The height over which the air thins by a factor of e. Air is not a uniform
// shell, and pretending it is has a visible cost: with constant density the
// optical depth saturates across almost the whole shell, so the halo is a flat
// saturated band with a hard outer edge -- the glass-marble look this shader
// exists to avoid. Thinning it with altitude makes the glow fade out well
// inside the shell, so the shell can stay tall enough to fly through while the
// rim stays tight.
uniform float scale_height = 5.0;
uniform float surface_radius = 100.0;
uniform float atmosphere_radius = 110.0;

// Distance along `dir` at which a ray from `origin` enters and leaves a sphere
// of `radius` centred on the origin of the world. Returns (enter, exit), or a
// zero-length span when the ray misses.
vec2 shell_span(vec3 origin, vec3 dir, float radius) {
	float b = dot(origin, dir);
	float c = dot(origin, origin) - radius * radius;
	float disc = b * b - c;
	if (disc <= 0.0) {
		return vec2(0.0);
	}
	float root = sqrt(disc);
	return vec2(-b - root, -b + root);
}

void fragment() {
	// The shell is a child sitting at the planet's own origin, so the model
	// matrix's translation is the planet's centre. Reading it here rather than
	// pushing a uniform every frame keeps a moving planet correct for free.
	vec3 centre = MODEL_MATRIX[3].xyz;
	vec3 camera = INV_VIEW_MATRIX[3].xyz;
	vec3 world_position = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
	vec3 dir = normalize(world_position - camera);
	vec3 origin = camera - centre;

	vec2 air = shell_span(origin, dir, atmosphere_radius);
	// Clamped at zero so a camera already inside the shell starts measuring
	// from where it is rather than from behind itself.
	float enter = max(air.x, 0.0);
	float leave = air.y;

	// Whatever the solid planet blocks is not air. This is what stops the
	// horizon being infinitely bright when you stand on the ground and look
	// along it: the ray hits dirt.
	vec2 body = shell_span(origin, dir, surface_radius);
	if (body.y > 0.0 && body.x > 0.0) {
		leave = min(leave, body.x);
	}

	float path = max(leave - enter, 0.0);

	// March the path and accumulate air, thinning with altitude. Twelve steps:
	// eight shows banding on the limb of the largest world, and past twelve
	// nothing about the picture changes.
	const int STEPS = 12;
	float step = path / float(STEPS);
	float column = 0.0;
	for (int i = 0; i < STEPS; i++) {
		vec3 sample_point = origin + dir * (enter + (float(i) + 0.5) * step);
		float altitude = max(length(sample_point) - surface_radius, 0.0);
		column += exp(-altitude / scale_height) * step;
	}
	// Normalised by the scale height, so a vertical ray from the ground is one
	// unit of air whatever size the planet is and `density` means the same
	// thing everywhere.
	float optical = 1.0 - exp(-column / scale_height * density);

	// Is that air lit? Sampled at the middle of the path rather than at the
	// fragment, so the term describes the air the ray actually crossed.
	vec3 midpoint = centre + origin + dir * (enter + path * 0.5);
	vec3 to_sun = normalize(sun_position - midpoint);
	vec3 up = normalize(midpoint - centre);
	// Wrapped rather than clamped at zero: the glow has to survive a little way
	// past the terminator, or the day-night edge is a hard line across the halo.
	float lit = clamp(dot(up, to_sun) * 0.6 + 0.4, 0.0, 1.0);

	// Grazing light reddens, which is the whole of what a sunrise is -- but only
	// where the light actually grazes, which is the band around the terminator
	// where the sun sits on the local horizon. Keying this off the angle between
	// the view and the sun instead painted the entire halo at sunset, because
	// from anywhere but straight down the sun line those two are nearly
	// perpendicular over most of a planet's rim.
	float twilight = 1.0 - abs(dot(up, to_sun));
	vec3 tint = mix(atmosphere_color, sunset_color,
		smoothstep(0.82, 1.0, twilight));

	ALBEDO = tint;
	ALPHA = optical * lit;
}
