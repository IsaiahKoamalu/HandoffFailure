extends StaticBody3D
class_name Station

## An orbiting structure with a docking port.
##
## Deliberately not a `Planet`. A Planet carries gravity, terrain, an ocean and
## a sphere of influence, and a station with all of those would be a fourth body
## pulling on the ship and — worse — a capture radius overlapping the planet it
## orbits, which is the one thing `get_dominant_planet` cannot survive (see the
## invariants in PLANETS.md). A station is scenery with a port on it.
##
## It borrows the one thing from Planet that matters: **the orbit is analytic**,
## a closed-form function of `SolarSystem.time` rather than an integration. Same
## reasons as there — exact time warp, no altitude drift, and a reload that
## reproduces the same state. It uses the same sign convention too, a negative
## `_mean_motion` running clockwise about `Vector3.UP`. If you change one,
## change both.

## The body this station orbits. Its position is read every tick, so the parent
## must already have been advanced — SolarSystem drives stations after planets.
@export var orbital_parent: Planet
@export var orbital_radius: float = 0.0
@export_range(0.0, 360.0, 0.1, "or_greater", "or_less") var orbital_angle: float = 0.0
## Seconds for one revolution. Independent of the parent's gravity on purpose: a
## station holds its orbit by whatever means a station holds its orbit, and
## tying the period to a circular-orbit formula would move it out of the useful
## band every time the planet was retuned.
@export var orbital_period: float = 240.0

@export_category("Docking")
## Where the port sits, in the station's own frame.
##
## At deck level, not at the hull's centreline. A ship's port is at its cabin
## floor, so putting this anywhere else means the two floors meet at a step —
## and the astronaut walks between them.
@export var port_offset: Vector3 = Vector3(0.0, -4.0, -14.0)
## Outward normal of the port face, in the station's own frame: the direction a
## ship approaches *from*. A docking ship travels along the negative of this.
@export var port_axis: Vector3 = Vector3(0.0, 0.0, -1.0)
## How far out along the normal the approach corridor begins.
@export var approach_range: float = 60.0
## Half-extents of the hull, in the station's own frame. Kept here rather than
## read off the mesh so a controller can ask the station how big it is without
## walking the scene, and so the box the physics uses and the box the reward
## uses cannot drift apart.
@export var hull_extents: Vector3 = Vector3(8.0, 5.0, 12.0)

@export_category("Interior")
## Half-extents of the walkable volume, in the station's own frame. Smaller than
## the hull because the hull has walls: this is the room inside them, and it is
## what decides whether an astronaut is aboard.
##
## It has to reach the *inner faces*, not stop short of them. Set a metre tight
## on the long axis, an astronaut who walked to the fore wall was standing on
## the deck, against the wall, and reported as not aboard -- so the frame would
## have handed them back to the planet while they were plainly inside the room.
## The walls sit at +/-7.5 and +/-11.5 with a half-metre skin, and the deck runs
## from -4.0 to +2.8; these are those numbers.
@export var interior_extents: Vector3 = Vector3(7.0, 3.4, 11.0)
## Centre of that volume, so a deck that is not halfway up the hull still works.
@export var interior_center: Vector3 = Vector3(0.0, -0.6, 0.0)
## Downward acceleration inside, along the station's own -Y.
##
## Spun-up, plated, handwaved — the game does not say which, the same way the
## ship's cabin does not. What matters is that it is a *frame*: the astronaut
## reparents into it and walks about while the whole room orbits a planet at
## eight units a second, and never has to know that.
@export var deck_gravity: float = 9.81
## The passage from the port to the room: half-extents and centre, in the
## station's own frame.
##
## Counted as interior, so an astronaut in the tunnel rides the station. It
## deliberately reaches well *past* the port, into where a docked ship's nose
## sits: `Astronaut._pick_frame` asks the ship first, so the overlap costs
## nothing, and without it there is a band where neither volume claims the
## doorway and the astronaut is handed back to the planet mid-step.
##
## How far past is not a guess. The latch tolerates two units of port-to-port
## error, so the join sits anywhere in a two-unit band and a volume sized for a
## perfect latch misses it. Measured before this was sized properly: a 0.26-unit
## gap the astronaut walked into and stopped dead in, flickering between the
## ship's frame and the star's twice a second.
@export var airlock_extents: Vector3 = Vector3(1.0, 1.0, 2.75)
@export var airlock_center: Vector3 = Vector3(0.0, -3.0, -13.75)
## The hatch panel closing the tunnel off from vacuum. Hidden and un-collided
## while a ship is docked, which is the whole of what "the airlock is open"
## means here.
@export var hatch: Node3D

# `constant_linear_velocity` is StaticBody3D's own field, written here the same
# way Planet writes it: it is what a body's own motion is read from everywhere
# in the project, so a station answers that question without special-casing.

var _mean_motion: float = 0.0
var _phase: float = 0.0


func init() -> void:
	_phase = deg_to_rad(orbital_angle)
	# Negative, so a station runs the same way round as everything else.
	_mean_motion = -TAU / maxf(orbital_period, 0.001)
	update_orbit(0.0)


## Position, velocity and attitude at a time, in closed form. Mirrors
## Planet.update_orbit including the sign of the tangent, so a station and a
## moon at the same radius travel the same way round.
func update_orbit(time: float) -> void:
	if orbital_parent == null:
		return
	var angle: float = _phase + _mean_motion * time
	var radial := Vector3(-sin(angle), 0.0, -cos(angle))
	var tangent := Vector3(-cos(angle), 0.0, sin(angle))

	global_position = orbital_parent.global_position + radial * orbital_radius
	constant_linear_velocity = orbital_parent.constant_linear_velocity \
		+ tangent * (_mean_motion * orbital_radius)

	# Held pointing rather than tumbling, with the port facing away from the
	# planet so a ship arriving from open space meets it head on instead of
	# having to fly around the structure. That is a choice about the task and
	# not about physics: matching attitude with a spinning target is a harder
	# problem than the one being posed first, and it can be turned on later by
	# spinning this basis without anything downstream changing.
	global_basis = Basis.looking_at(-radial, Vector3.UP)


# --- The port ----------------------------------------------------------------

## World position of the docking port.
func port_position() -> Vector3:
	return global_position + global_basis * port_offset


## World-space outward normal of the port face. A ship is lined up when its own
## port normal points the opposite way — the two faces meet, so the dot product
## of the normals is -1, not +1.
func port_normal() -> Vector3:
	return (global_basis * port_axis).normalized()


# --- Interior ----------------------------------------------------------------

## True when a world point is inside the walkable volume. Same shape and same
## job as Ship.contains_point, because the astronaut asks both the same
## question when it decides what it is standing in.
func contains_point(world_pos: Vector3) -> bool:
	var local: Vector3 = global_basis.inverse() * (world_pos - global_position)
	return _within(local, interior_center, interior_extents) \
		or _within(local, airlock_center, airlock_extents)


static func _within(local: Vector3, centre: Vector3, extents: Vector3) -> bool:
	var d: Vector3 = local - centre
	return absf(d.x) <= extents.x and absf(d.y) <= extents.y and absf(d.z) <= extents.z


## Open or close the airlock hatch. Called by the ship as it latches and lets
## go, so the passage exists exactly while there is something on the other side
## of it.
func set_hatch_open(open: bool) -> void:
	if hatch == null:
		return
	hatch.visible = not open
	for child in hatch.get_children():
		if child is CollisionShape3D:
			(child as CollisionShape3D).set_deferred("disabled", open)
	if hatch is CollisionShape3D:
		(hatch as CollisionShape3D).set_deferred("disabled", open)


## World-space velocity of the deck at a point, matching Planet's and Ship's
## signature so the astronaut can convert frames without caring which kind of
## body it is riding. The station does not spin, so every point moves alike.
func get_surface_velocity(_world_pos: Vector3) -> Vector3:
	return constant_linear_velocity


## The ship currently latched here, if any. Walked rather than cached because
## docking is rare and a stale reference to a freed ship is a crash.
func docked_ship() -> Ship:
	for client in SolarSystem.Instance.get_terrain_clients():
		if client is Ship and (client as Ship).docked_to == self:
			return client as Ship
	return null


## Airlock control: climb back into whatever is docked here.
func _on_board_activated(_who: Node3D) -> void:
	var ship := docked_ship()
	if ship != null:
		ship.board()


## Velocity of the port itself. Equal to the station's, because the station does
## not spin — but callers should ask rather than assume, so that giving a station
## rotation later changes this and nothing else.
func port_velocity() -> Vector3:
	return constant_linear_velocity


## True when a world point is inside the hull, inflated by `margin`.
##
## A box test rather than a sphere: the station is 16 by 10 by 24 and its port
## sits two units off the short face, so a sphere large enough to contain the
## hull would also contain a correctly docked ship and score every success as a
## collision.
func is_inside_hull(world_pos: Vector3, margin: float = 0.0) -> bool:
	var local: Vector3 = global_basis.inverse() * (world_pos - global_position)
	return absf(local.x) < hull_extents.x + margin \
		and absf(local.y) < hull_extents.y + margin \
		and absf(local.z) < hull_extents.z + margin


## Where a ship begins its final run: out along the port normal by
## `approach_range`. Guidance aims here first, so a ship lines up with the
## corridor before it starts closing on the port.
func approach_point() -> Vector3:
	return port_position() + port_normal() * approach_range
