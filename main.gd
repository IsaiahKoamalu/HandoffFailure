extends Node3D
class_name SolarSystem

static var Instance: SolarSystem
var Planets: Array[Planet] = []

func _ready() -> void:
	Instance = self

	# Collect children that are Planet instances
	Planets.clear()
	for child in get_children():
		if child is Planet:
			Planets.append(child)

	# Order planets by how deep they are in the hierarchy (fewest parents first)
	Planets.sort_custom(func(a: Planet, b: Planet) -> bool:
		return a.how_many_parents() < b.how_many_parents()
	)

	# Initialize in parent-first order
	for planet in Planets:
		planet.init()
		

# Button to exit the sim
func _on_button_quit_pressed() -> void:
	get_tree().quit()
