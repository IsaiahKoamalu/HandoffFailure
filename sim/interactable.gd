extends Area3D
class_name Interactable

## Something an Astronaut can walk up to and activate with the interact key.
##
## Interactables sit on their own collision layer and are picked up by the
## astronaut's InteractProbe, so adding a new one is purely a scene-level job:
## drop in the node, set a prompt, and connect `activated`.

## Emitted when an astronaut in range presses interact.
signal activated(who: Node3D)

## Verb shown in the HUD prompt, e.g. "Fly ship".
@export var prompt: String = "Interact"
## Switched off when the action does not currently apply.
@export var enabled: bool = true


func activate(who: Node3D) -> void:
	if enabled:
		activated.emit(who)
