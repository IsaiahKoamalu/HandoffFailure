extends AudioStreamPlayer3D
class_name EngineAudio

## Engine rumble, synthesised rather than sampled.
##
## Nothing here needs an audio file: the stream is filled each frame with a few
## detuned harmonics plus filtered noise, pitched and levelled by the ship's
## throttle. It is a placeholder with the right behaviour — swap `stream` for a
## looping .ogg and delete `_fill()` when you have a real recording, keeping the
## throttle mapping below.

const MIX_RATE := 22050.0

@export var _ship: Ship
## Fundamental at idle, in Hz. The engine rides up from here under thrust.
@export var _idle_hz: float = 38.0
@export_range(0.0, 2.0, 0.05) var _pitch_climb: float = 0.55
## Kept audible at rest so the ship never sounds dead.
@export_range(0.0, 0.5, 0.01) var _idle_level: float = 0.09
@export_range(0.0, 1.0, 0.01) var _full_level: float = 0.62
## How quickly the note follows the throttle. Slower than the throttle itself,
## so the engine sounds like it is catching up rather than snapping.
@export_range(0.5, 20.0, 0.1) var _response: float = 4.0

var _playback: AudioStreamGeneratorPlayback
var _phase: float = 0.0
var _level: float = 0.0
var _noise: float = 0.0
var _rng := RandomNumberGenerator.new()


func _ready() -> void:
	var generator := AudioStreamGenerator.new()
	generator.mix_rate = MIX_RATE
	generator.buffer_length = 0.12
	stream = generator
	autoplay = false
	play()
	_playback = get_stream_playback()


func _process(delta: float) -> void:
	if _playback == null or _ship == null:
		return

	_level = lerp(_level, _ship.throttle, clampf(_response * delta, 0.0, 1.0))

	var amplitude: float = lerp(_idle_level, _full_level, _level)
	volume_db = linear_to_db(maxf(amplitude, 0.0001))

	var hz: float = _idle_hz * (1.0 + _pitch_climb * _level)
	var frames: int = _playback.get_frames_available()
	for i in frames:
		_phase = fmod(_phase + hz / MIX_RATE, 1.0)
		var angle: float = TAU * _phase
		# A few harmonics rather than a pure tone, or it sounds like a test bench.
		var tone: float = (sin(angle) * 0.55
			+ sin(angle * 2.01) * 0.24
			+ sin(angle * 3.02) * 0.11)
		# One-pole low pass on white noise: the roar under the note.
		_noise = lerp(_noise, _rng.randf_range(-1.0, 1.0), 0.14)
		var sample: float = tone * 0.7 + _noise * (0.30 + 0.35 * _level)
		_playback.push_frame(Vector2(sample, sample) * 0.5)


func _exit_tree() -> void:
	# Release the playback so the generator is not left referenced at shutdown.
	_playback = null
