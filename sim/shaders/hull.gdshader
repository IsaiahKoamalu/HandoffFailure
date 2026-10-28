shader_type spatial;
render_mode diffuse_burley, specular_schlick_ggx;

// Panelled metal for the ship and the station.
//
// Both are built out of untextured boxes, and a box with one flat albedo has no
// size: a two-metre hull panel and a twenty-metre station wall look identical
// because there is nothing on either to count. Panel seams give them a unit,
// and the seams are laid out in OBJECT space so a long wall gets many panels
// and a short one gets few, without anybody authoring UVs for either.
//
// Everything here is arithmetic on position. No textures, because the whole
// point is that these meshes have no texture coordinates to sample one with.

uniform vec3 hull_color : source_color = vec3(0.42, 0.45, 0.50);
uniform float metallic_amount : hint_range(0.0, 1.0) = 0.55;
uniform float roughness_amount : hint_range(0.0, 1.0) = 0.55;

// Panel size in object-space units, and how dark the seams between them cut.
uniform float panel_size = 0.85;
uniform float seam_width : hint_range(0.001, 0.2) = 0.028;
uniform float seam_depth : hint_range(0.0, 1.0) = 0.45;
// How much neighbouring panels differ from each other. Small: this is meant to
// read as weathering and replaced plating, not as a chequerboard.
uniform float panel_variation : hint_range(0.0, 0.5) = 0.06;

// A lit strip along one axis, for running lights and door surrounds.
uniform vec3 trim_color : source_color = vec3(0.35, 0.7, 1.0);
uniform float trim_strength : hint_range(0.0, 4.0) = 0.0;
// Position and half-thickness of the strip along object-space Y.
uniform float trim_height = 0.0;
uniform float trim_thickness = 0.06;

varying vec3 object_position;
varying vec3 object_normal;

void vertex() {
	object_position = VERTEX;
	object_normal = NORMAL;
}

float hash21(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Distance to the nearest seam, projected onto whichever pair of axes the
// surface faces. Projecting rather than sampling three times is what keeps this
// from smearing the grid across a face that is nearly edge-on.
vec2 panel_plane(vec3 p, vec3 n) {
	vec3 a = abs(n);
	if (a.x > a.y && a.x > a.z) {
		return p.zy;
	}
	if (a.y > a.z) {
		return p.xz;
	}
	return p.xy;
}

void fragment() {
	vec3 n = normalize(object_normal);
	vec2 plane = panel_plane(object_position, n) / panel_size;
	vec2 cell = floor(plane);
	vec2 within = abs(fract(plane) - 0.5);

	// The seam is the last sliver of a cell before the next one starts.
	float edge = max(within.x, within.y);
	float seam = smoothstep(0.5 - seam_width / panel_size, 0.5, edge);

	float panel = (hash21(cell) - 0.5) * 2.0 * panel_variation;
	vec3 base = hull_color * (1.0 + panel) * (1.0 - seam * seam_depth);

	ALBEDO = base;
	METALLIC = metallic_amount;
	// Seams are recessed and hold dirt, so they are rougher than the plate.
	ROUGHNESS = mix(roughness_amount, min(roughness_amount + 0.35, 1.0), seam);
	SPECULAR = 0.5;

	float strip = 1.0 - smoothstep(0.0, trim_thickness,
		abs(object_position.y - trim_height));
	EMISSION = trim_color * trim_strength * strip;
}
