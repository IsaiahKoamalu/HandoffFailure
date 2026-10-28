extends StaticBody3D
class_name Landmark

## Something on a planet's surface worth flying to.
##
## The worlds have distinct terrain, weather and ground detail, and none of that
## gives a pilot a reason to prefer one landing site over another. These do: a
## handful of sites per planet, each visible from the air, each targetable from
## the navigation panel, and each with something to walk up to and survey.
##
## Placed by the planet from its own seed, as children of it, so they ride the
## surface through the planet's spin exactly as terrain does. Nothing about them
## is saved except which ones have been visited -- position is a pure function
## of the seed, like everything else on these worlds.

enum Kind {
	## A tall mast with a light on it. The one you can see from orbit.
	BEACON,
	## A ship that did not land as well as yours.
	WRECK,
	## A mineral spire. Lights its own ground.
	OUTCROP,
}

## Type decides silhouette, colour and prompt. Nothing else varies.
var kind: Kind = Kind.BEACON
## The planet this stands on. Set before the node enters the tree.
var planet: Planet = null
## Surveyed by the astronaut at least once.
var discovered: bool = false
## Stable across runs and saves: the planet's seed and the site's index.
var site_id: String = ""

var _interactable: Interactable
var _light: OmniLight3D


static func kind_name(which: Kind) -> String:
	match which:
		Kind.BEACON: return "Beacon"
		Kind.WRECK: return "Wreck"
		Kind.OUTCROP: return "Outcrop"
	return "Site"


## Display name, which is what the navigation panel and the HUD show.
func label() -> String:
	return "%s %s" % [kind_name(kind), site_id]


func _ready() -> void:
	# Layer 1 is the world: the astronaut's mask includes it, so these are solid
	# to walk into rather than scenery to walk through.
	collision_layer = 1
	collision_mask = 0
	_build()


func _build() -> void:
	match kind:
		Kind.BEACON:
			_build_beacon()
		Kind.WRECK:
			_build_wreck()
		Kind.OUTCROP:
			_build_outcrop()

	_interactable = Interactable.new()
	_interactable.name = "Survey"
	_interactable.prompt = "[E] Survey %s" % kind_name(kind).to_lower()
	_interactable.collision_layer = 4
	_interactable.collision_mask = 0
	var reach := CollisionShape3D.new()
	var sphere := SphereShape3D.new()
	sphere.radius = 4.0
	reach.shape = sphere
	_interactable.add_child(reach)
	_interactable.position = Vector3(0.0, 1.5, 0.0)
	_interactable.activated.connect(_on_surveyed)
	add_child(_interactable)


func _mesh(mesh: Mesh, at: Vector3, basis: Basis = Basis.IDENTITY) -> void:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.transform = Transform3D(basis, at)
	add_child(instance)


func _shape(shape: Shape3D, at: Vector3, basis: Basis = Basis.IDENTITY) -> void:
	var collider := CollisionShape3D.new()
	collider.shape = shape
	collider.transform = Transform3D(basis, at)
	add_child(collider)


func _metal(color: Color, emission: Color = Color(0, 0, 0, 0)) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.metallic = 0.55
	mat.roughness = 0.5
	if emission.a > 0.0:
		mat.emission_enabled = true
		mat.emission = emission
		mat.emission_energy_multiplier = 4.0
	return mat


## A mast with a light on top. Tall on purpose: this is the one that has to be
## findable from the air, and eight metres of mast against open ground reads
## from a long way further off than anything sitting on the surface.
func _build_beacon() -> void:
	var mast := CylinderMesh.new()
	mast.top_radius = 0.18
	mast.bottom_radius = 0.3
	mast.height = 8.0
	mast.radial_segments = 8
	mast.material = _metal(Color(0.55, 0.57, 0.6))
	_mesh(mast, Vector3(0.0, 4.0, 0.0))

	var mast_shape := CylinderShape3D.new()
	mast_shape.radius = 0.3
	mast_shape.height = 8.0
	_shape(mast_shape, Vector3(0.0, 4.0, 0.0))

	var base := BoxMesh.new()
	base.size = Vector3(2.4, 0.4, 2.4)
	base.material = _metal(Color(0.4, 0.42, 0.46))
	_mesh(base, Vector3(0.0, 0.2, 0.0))
	var base_shape := BoxShape3D.new()
	base_shape.size = Vector3(2.4, 0.4, 2.4)
	_shape(base_shape, Vector3(0.0, 0.2, 0.0))

	var lamp := SphereMesh.new()
	lamp.radius = 0.42
	lamp.height = 0.84
	lamp.radial_segments = 10
	lamp.rings = 6
	lamp.material = _metal(Color(0.9, 0.35, 0.2), Color(1.0, 0.4, 0.15))
	_mesh(lamp, Vector3(0.0, 8.2, 0.0))

	_light = OmniLight3D.new()
	_light.position = Vector3(0.0, 8.2, 0.0)
	_light.light_color = Color(1.0, 0.5, 0.2)
	_light.light_energy = 4.0
	_light.omni_range = 26.0
	add_child(_light)


## A hull that arrived faster than it meant to: a tilted slab, a broken spar and
## some scattered plate. Deliberately the same grey as the player's ship.
func _build_wreck() -> void:
	var hull := BoxMesh.new()
	hull.size = Vector3(3.6, 1.6, 6.0)
	hull.material = _metal(Color(0.36, 0.38, 0.42))
	var tilt := Basis(Vector3.RIGHT, deg_to_rad(-24.0)).rotated(Vector3.UP, 0.4)
	_mesh(hull, Vector3(0.0, 1.0, 0.0), tilt)
	var hull_shape := BoxShape3D.new()
	hull_shape.size = Vector3(3.6, 1.6, 6.0)
	_shape(hull_shape, Vector3(0.0, 1.0, 0.0), tilt)

	var spar := CylinderMesh.new()
	spar.top_radius = 0.16
	spar.bottom_radius = 0.16
	spar.height = 5.0
	spar.radial_segments = 6
	spar.material = _metal(Color(0.3, 0.31, 0.34))
	_mesh(spar, Vector3(2.0, 1.4, -1.0),
		Basis(Vector3.FORWARD, deg_to_rad(64.0)))

	var plate := BoxMesh.new()
	plate.size = Vector3(1.4, 0.2, 1.8)
	plate.material = _metal(Color(0.33, 0.34, 0.38))
	_mesh(plate, Vector3(-3.2, 0.15, 2.4), Basis(Vector3.UP, 1.1))
	_mesh(plate, Vector3(3.9, 0.12, -3.1), Basis(Vector3.UP, -0.6))


## A mineral spire that lights the ground under it. The only one of the three
## that is not obviously man-made, and the only one worth looking at up close
## rather than from the air.
func _build_outcrop() -> void:
	var glow := Color(0.35, 0.85, 1.0)
	for i in 5:
		var height: float = 2.2 + i * 0.9
		var spike := CylinderMesh.new()
		spike.top_radius = 0.05
		spike.bottom_radius = 0.35 + i * 0.06
		spike.height = height
		spike.radial_segments = 6
		spike.material = _metal(Color(0.2, 0.34, 0.42), glow)
		var angle: float = i * TAU / 5.0
		var offset := Vector3(cos(angle) * 1.3, height * 0.5, sin(angle) * 1.3)
		var lean := Basis(Vector3(sin(angle), 0.0, -cos(angle)), deg_to_rad(11.0))
		_mesh(spike, offset, lean)

	var rock := SphereShape3D.new()
	rock.radius = 2.0
	_shape(rock, Vector3(0.0, 1.0, 0.0))

	_light = OmniLight3D.new()
	_light.position = Vector3(0.0, 2.0, 0.0)
	_light.light_color = glow
	_light.light_energy = 2.4
	_light.omni_range = 18.0
	add_child(_light)


func _on_surveyed(_who: Node3D) -> void:
	if discovered:
		return
	discovered = true
	_interactable.prompt = "%s surveyed" % kind_name(kind)
	print("Surveyed %s on %s" % [label(), planet.label() if planet != null else "?"])
