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
@export var port_offset: Vector3 = Vector3(0.0, 0.0, -14.0)
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
