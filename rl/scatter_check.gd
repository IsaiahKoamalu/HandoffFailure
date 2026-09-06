extends Node3D

## Do the boulders stay where they were put?
##
##     Godot.exe --fixed-fps 60 --path . res://rl/scatter_check.tscn
##
## Not headless: scatter is skipped without a renderer, because it is decoration
## and nothing that runs headless can see it.
##
## The failure this exists to catch is that boulders wink out and reappear
## somewhere else as you walk -- which is what happens if their placement is
## hashed by their index in the streaming patch's grid rather than by where they
## are on the planet. It is invisible in a screenshot and unmistakable in motion,
## so the test has to compare two DIFFERENT patch centres over the ground they
## share. Streaming away and back to a byte-identical centre reproduces even a
## patch-local grid, and a check that only did that would pass on the bug.

const MAIN_SCENE := "res://main.tscn"
## Fraction of a patch to shift by. Small enough that the two patches overlap
## heavily, large enough that a patch-local grid lands somewhere else entirely.
const SHIFT: float = 0.25
## Minimum boulders in the overlap for the comparison to mean anything. Low,
## because the sparsest worlds put very little loose rock on the ground and the
## check still has to say something about them.
const MIN_OVERLAP: int = 8

var _client := Node3D.new()
var _failures: int = 0


func _ready() -> void:
	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)
	add_child(_client)

	var tested: int = 0
	for planet in SolarSystem.Instance.Planets:
		if not planet.terrain_enabled or not planet.scatter_enabled:
			continue
		tested += 1
		_check(planet)
	if tested == 0:
		print("RESULT no_scatter")
		get_tree().quit(1)
		return
	print("RESULT %s (%d failures over %d worlds)" % [
		"pass" if _failures == 0 else "FAIL", _failures, tested])
	get_tree().quit(0 if _failures == 0 else 1)


func _check(planet: Planet) -> void:
	var home: Vector3 = planet.largest_landmass_center
	var axis: Vector3 = home.cross(Vector3.UP).normalized()
	var patch: float = planet._patch_angle()

	var first := _sample(planet, home)
	var second := _sample(planet, home.rotated(axis, patch * SHIFT).normalized())

	# Only well inside both patches, so a boulder missing because it fell off
	# an edge does not count as one that moved.
	# 0.6 of the patch, shifted by 0.25 of it, stays inside the 0.92 the scatter
	# is sampled within -- so everything counted here is ground both patches
	# actually cover.
	var core: float = cos(patch * 0.6)
	var checked: int = 0
	var moved: int = 0
	for key in first:
		if first[key].normalized().dot(home) < core:
			continue
		checked += 1
		if not second.has(key):
			moved += 1

	var ok: bool = checked >= MIN_OVERLAP and moved == 0
	if not ok:
		_failures += 1
	print("%-8s %s  %d in both patches, %d moved  (patches held %d and %d)" % [
		planet.name, "ok  " if ok else "FAIL", checked, moved,
		first.size(), second.size()])


## Boulder positions in planet-local space for a patch centred on `dir`, keyed
## by rounded position so the comparison does not depend on instance order.
func _sample(planet: Planet, dir: Vector3) -> Dictionary:
	var at: Vector3 = planet.global_position + planet.global_basis * (
		dir * (planet.get_surface_radius_at(dir) + 2.0))
	_client.global_position = at
	planet.build_collision_now(_client, at)
	var out: Dictionary = {}
	for child in planet.get_children():
		var mm := child as MultiMeshInstance3D
		if mm == null or not mm.name.begins_with("Scatter_"):
			continue
		for i in mm.multimesh.instance_count:
			var p: Vector3 = mm.multimesh.get_instance_transform(i).origin
			out[Vector3i(roundi(p.x * 8.0), roundi(p.y * 8.0), roundi(p.z * 8.0))] = p
	return out
