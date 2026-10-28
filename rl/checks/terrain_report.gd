extends Node3D

## Measures the shape of every planet's terrain, so "too spiky" and "mostly
## ocean" can be numbers instead of impressions.
##
##     Godot.exe --headless --fixed-fps 60 --path . res://rl/checks/terrain_report.tscn
##
## Reports, per planet:
##
##   land        fraction of the sphere above sea level
##   biggest     the largest single landmass, as a fraction of the sphere
##   relief      mean and peak terrain height, in world units
##   slope       mean and 99th-percentile ground slope in degrees, sampled at
##               the scale a landing ship actually cares about
##
## Slope is the one that matters for spikiness. Height alone does not
## distinguish a mountain range from a bed of nails; slope over a fixed ground
## distance does, and it is also exactly what decides whether a set of landing
## struts can sit level on the result.

const MAIN_SCENE := "res://sim/main.tscn"
## Directions sampled per planet. Fibonacci-distributed, so this is an even
## covering rather than a random one and the numbers are stable run to run.
const SAMPLES: int = 20000
## Ground distance between the pair of points a slope is measured over, in
## world units. Roughly a ship length: the question is whether the ground is
## flat enough to set down on, not whether the planet is smooth from orbit.
const SLOPE_BASELINE: float = 8.0
## Directions used for the landmass flood fill. Coarser than SAMPLES because it
## needs neighbour lookups, and a continent is a big target.
const PATCH_SAMPLES: int = 4000


func _ready() -> void:
	var world: Node = load(MAIN_SCENE).instantiate()
	add_child(world)

	for planet in SolarSystem.Instance.Planets:
		if not planet.terrain_enabled:
			continue
		_report(planet)
	get_tree().quit(0)


func _report(planet: Planet) -> void:
	var directions := _fibonacci_sphere(SAMPLES)

	var land: int = 0
	var height_total: float = 0.0
	var height_peak: float = -INF
	var slopes := PackedFloat32Array()

	# Angular offset that corresponds to SLOPE_BASELINE on the ground.
	var step: float = SLOPE_BASELINE / planet.surface_radius

	for dir in directions:
		var height: float = planet.get_terrain_height(dir)
		if height > 0.0:
			land += 1
			height_total += height
			height_peak = maxf(height_peak, height)
			slopes.append(_slope_degrees(planet, dir, step))

	if land == 0:
		print("%-8s  no land at all" % planet.name)
		return

	slopes.sort()
	var mean_slope: float = 0.0
	for value in slopes:
		mean_slope += value
	mean_slope /= float(slopes.size())
	var p99: float = slopes[mini(int(slopes.size() * 0.99), slopes.size() - 1)]

	print("%-8s  land %5.1f%%  biggest %5.1f%%  relief mean %6.2f peak %6.2f  slope mean %5.1f deg  p99 %5.1f deg"
		% [
			planet.name,
			100.0 * float(land) / float(SAMPLES),
			100.0 * _largest_landmass(planet),
			height_total / float(land),
			height_peak,
			mean_slope,
			p99,
		])


## Ground slope at a direction, in degrees, measured over SLOPE_BASELINE.
##
## Takes the steeper of two perpendicular tangents rather than averaging them:
## a ridge is steep across and flat along, and averaging would call it gentle.
func _slope_degrees(planet: Planet, dir: Vector3, step: float) -> float:
	var east: Vector3 = dir.cross(Vector3.UP)
	if east.length_squared() < 1e-8:
		east = dir.cross(Vector3.RIGHT)
	east = east.normalized()
	var north: Vector3 = east.cross(dir).normalized()

	var here: float = planet.get_surface_radius_at(dir)
	var worst: float = 0.0
	for tangent in [east, north]:
		var ahead: float = planet.get_surface_radius_at((dir + tangent * step).normalized())
		worst = maxf(worst, absf(ahead - here))
	return rad_to_deg(atan2(worst, SLOPE_BASELINE))


## Fraction of the sphere taken by the single biggest connected landmass.
##
## Total land area does not say whether it is one continent or ten thousand
## islands, and that is the whole difference between a planet and static.
func _largest_landmass(planet: Planet) -> float:
	var directions := _fibonacci_sphere(PATCH_SAMPLES)
	var is_land := PackedInt32Array()
	is_land.resize(PATCH_SAMPLES)
	for i in PATCH_SAMPLES:
		is_land[i] = 1 if planet.get_terrain_height(directions[i]) > 0.0 else 0

	# Points are evenly spaced, so a neighbour is anything within a little over
	# the nominal spacing. N points over a sphere of area 4*PI leaves each with
	# 4*PI/N, hence a spacing of the square root of that; 1.5x reaches the
	# immediate ring without jumping a one-point channel.
	var reach: float = 1.5 * sqrt(4.0 * PI / float(PATCH_SAMPLES))
	var reach_squared: float = reach * reach

	var seen := PackedInt32Array()
	seen.resize(PATCH_SAMPLES)
	var best: int = 0

	for start in PATCH_SAMPLES:
		if seen[start] == 1 or is_land[start] == 0:
			continue
		var size: int = 0
		var stack: Array[int] = [start]
		seen[start] = 1
		while not stack.is_empty():
			var current: int = stack.pop_back()
			size += 1
			for other in PATCH_SAMPLES:
				if seen[other] == 1 or is_land[other] == 0:
					continue
				if directions[current].distance_squared_to(directions[other]) <= reach_squared:
					seen[other] = 1
					stack.append(other)
		best = maxi(best, size)

	return float(best) / float(PATCH_SAMPLES)


## Evenly spaced points on a sphere, via the golden angle.
static func _fibonacci_sphere(count: int) -> Array[Vector3]:
	var points: Array[Vector3] = []
	points.resize(count)
	var golden: float = PI * (3.0 - sqrt(5.0))
	for i in count:
		var y: float = 1.0 - 2.0 * (float(i) + 0.5) / float(count)
		var radius: float = sqrt(maxf(1.0 - y * y, 0.0))
		var theta: float = golden * float(i)
		points[i] = Vector3(cos(theta) * radius, y, sin(theta) * radius)
	return points
