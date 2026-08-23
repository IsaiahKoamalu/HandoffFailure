class_name PlanetMesh

## Builds cube-sphere geometry from a Planet's height function.
##
## Six subdivided planes are pushed out onto the sphere, so quads stay roughly
## uniform everywhere instead of collapsing at the poles the way a UV sphere's
## do. Every vertex is placed by Planet.get_surface_radius_at, so the render
## mesh and the collision patches can never disagree about where the ground is.
##
## Everything here works in planet-LOCAL space. That matters: terrain is fixed
## to the surface, so it has to rotate with the planet's spin rather than stay
## put in world space.

const FACE_NORMALS: Array[Vector3] = [
	Vector3.UP, Vector3.DOWN, Vector3.LEFT,
	Vector3.RIGHT, Vector3.FORWARD, Vector3.BACK,
]


## Full-planet mesh. `resolution` is vertices per cube-face edge.
##
## Pass `fixed_radius` to build a smooth sphere at that radius instead of
## sampling the planet's terrain — that is how the ocean surface is built, and
## it accepts a null `planet` in that mode.
static func build(planet: Planet, resolution: int, fixed_radius: float = -1.0) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	for normal in FACE_NORMALS:
		_add_face(st, planet, normal, resolution, fixed_radius)
	st.generate_normals()
	# No generate_tangents(): it errors with "UVs are required to generate
	# tangents", and the triplanar material does not need them anyway.
	return st.commit()


static func _add_face(st: SurfaceTool, planet: Planet, normal: Vector3, resolution: int, fixed_radius: float) -> void:
	# Any two axes perpendicular to the face normal work; rotating the
	# components like this picks a consistent pair for all six faces.
	var axis_a := Vector3(normal.y, normal.z, normal.x)
	var axis_b := normal.cross(axis_a)

	var verts := PackedVector3Array()
	verts.resize(resolution * resolution)
	# Terrain carries per-vertex colour; the fixed-radius ocean surface does not,
	# and a SurfaceTool surface must be consistent about which attributes it has.
	var tinted := fixed_radius <= 0.0
	var cols := PackedColorArray()
	if tinted:
		cols.resize(resolution * resolution)

	for y in resolution:
		for x in resolution:
			var u := 2.0 * x / (resolution - 1.0) - 1.0
			var v := 2.0 * y / (resolution - 1.0) - 1.0
			var dir := _spherify(normal + axis_a * u + axis_b * v)
			var r := fixed_radius
			if tinted:
				var h := planet.get_terrain_height(dir)
				r = planet.surface_radius + h
				cols[y * resolution + x] = planet.get_terrain_color(h)
			verts[y * resolution + x] = dir * r

	for y in resolution - 1:
		for x in resolution - 1:
			var i := y * resolution + x

			# Godot treats clockwise as front-facing, the opposite of the
			# right-hand rule. This order is the one that ends up outward;
			# swapping the last two of each triangle turns the planet
			# inside out.
			_emit(st, verts, cols, tinted, i)
			_emit(st, verts, cols, tinted, i + resolution + 1)
			_emit(st, verts, cols, tinted, i + 1)

			_emit(st, verts, cols, tinted, i)
			_emit(st, verts, cols, tinted, i + resolution)
			_emit(st, verts, cols, tinted, i + resolution + 1)


# set_color must be called before add_vertex, and consistently across the
# whole surface.
static func _emit(st: SurfaceTool, verts: PackedVector3Array, cols: PackedColorArray, tinted: bool, i: int) -> void:
	if tinted:
		st.set_color(cols[i])
	st.add_vertex(verts[i])


## Collision mesh for a square patch of terrain centred on the planet-local
## direction `dir`. `half_angle` is the angular half-width in radians, so the
## patch scales with the planet instead of needing retuning per body.
##
## Returned in planet-local space, so the CollisionShape3D holding it must sit
## at the Planet's origin with an identity transform.
static func build_patch(planet: Planet, dir: Vector3, half_angle: float, resolution: int) -> ConcavePolygonShape3D:
	var up := dir.normalized()

	# Any tangent will do, but UP is degenerate at the planet's own poles.
	var axis_a := up.cross(Vector3.UP)
	if axis_a.length_squared() < 0.001:
		axis_a = up.cross(Vector3.RIGHT)
	axis_a = axis_a.normalized()
	var axis_b := up.cross(axis_a)

	var verts := PackedVector3Array()
	verts.resize(resolution * resolution)
	for y in resolution:
		for x in resolution:
			var u := (2.0 * x / (resolution - 1.0) - 1.0) * half_angle
			var v := (2.0 * y / (resolution - 1.0) - 1.0) * half_angle
			var d := (up + axis_a * u + axis_b * v).normalized()
			verts[y * resolution + x] = d * planet.get_surface_radius_at(d)

	var faces := PackedVector3Array()
	faces.resize((resolution - 1) * (resolution - 1) * 6)
	var f := 0
	for y in resolution - 1:
		for x in resolution - 1:
			var i := y * resolution + x
			faces[f] = verts[i]
			faces[f + 1] = verts[i + resolution + 1]
			faces[f + 2] = verts[i + 1]
			faces[f + 3] = verts[i]
			faces[f + 4] = verts[i + resolution]
			faces[f + 5] = verts[i + resolution + 1]
			f += 6

	var shape := ConcavePolygonShape3D.new()
	shape.set_faces(faces)
	return shape


# Maps a point on the [-1, 1] cube onto the sphere with far less corner
# bunching than plain normalization gives.
static func _spherify(p: Vector3) -> Vector3:
	var x2 := p.x * p.x
	var y2 := p.y * p.y
	var z2 := p.z * p.z
	return Vector3(
		p.x * sqrt(1.0 - (y2 + z2) * 0.5 + y2 * z2 / 3.0),
		p.y * sqrt(1.0 - (z2 + x2) * 0.5 + z2 * x2 / 3.0),
		p.z * sqrt(1.0 - (x2 + y2) * 0.5 + x2 * y2 / 3.0)
	).normalized()
