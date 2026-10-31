# The bridge protocol

Godot knows the physics. Python knows how to learn. Neither can do the other's job, so
they run as two processes and pass notes. This file is what the notes look like.

Write it down first because both sides have to agree. A field added in Python that the
engine does not send is a crash; a field whose *meaning* drifts between the two sides is
worse, because it trains.

## Shape of the conversation

Godot runs headless and listens on `127.0.0.1:11008`; Python connects to it. One message
per line, newline-delimited JSON, no newlines inside a message. The two sides strictly
alternate: the trainer sends one request, the engine sends exactly one reply, and the
engine never speaks unprompted.

Every message in both directions carries a `"type"`. A request the engine cannot handle
gets `{"type":"error","reason":"..."}` back instead of silence, and the trainer raises
on it. Silence is what a hang looks like, and a hang is the hardest failure here to
diagnose.

One exchange carries all 32 ships, because a round trip costs the same whether it
carries one ship or thirty-two.

One exchange is one **decision**, not one physics tick. The engine holds each action for
`repeat` ticks and sums the reward over them. The trainer sees one transition either
way; the ship just gets more simulated time per decision. While it waits for the next
request the engine blocks, so physics only advances when the trainer asks and a run is
reproducible from its seed.

## 1. Hello — once, first

The trainer says how the run is configured. The engine builds the world to that
configuration and describes the task it is running. The trainer does not assume any of
it.

```json
{"type":"hello","seed":7,"ships":32,"repeat":6,"max_steps":27000}
```

The configuration goes in this direction, not the other, so that one process decides it
and the training command records it. `max_steps` is optional and defaults to 27,000.

```json
{"type":"hello","task":"landing","ships":32,"repeat":6,"max_steps":27000,
 "obs_size":20,"act_size":6,
 "obs_names":["altitude","vertical_speed","horizontal_speed","fuel","..."],
 "outcomes":{"0":"running","1":"landed","2":"crashed","3":"escaped","4":"timeout"},
 "success":1,
 "planets":["Sun","Planet1","Planet2","Planet3","Planet4","Moon1"]}
```

The reply reports what was actually built, not what was asked for. On receiving it the
trainer asserts that `ships`, `repeat` and `max_steps` are what it asked for, and that
`obs_size` matches the policy it is about to load. A silent mismatch here is a policy
flying a task it was not trained for, which looks like bad learning and is not.

A second `hello` is an error. Building a second solar system beside the first would
double the gravity and the planets without complaint.

**`obs_names` is not decoration.** It is the one field with nothing to do with training
and everything to do with the project. E3 reports how far each input channel sat from
normal at the handoff, and that output has to read `vertical_speed sat 4.2 from normal`,
not `channel 11 sat 4.2 from normal`. Save these names alongside the weights. Nothing
else recovers them later.

`outcomes` keeps the names of the endings on the engine side only. A second copy in
Python is a copy that drifts. Its keys are strings because JSON object keys have to be.

## 2. Step — the loop

Trainer sends one action row per ship. Six floats: the ship's thrust and turn axes, each
in [−1, 1]. The engine clamps them to that range; the trainer still computes
log-probabilities on what it sampled.

```json
{"type":"step","actions":[[0.1,-0.4,0.0,0.0,0.2,0.0], "...32 rows..."]}
```

Engine replies, `repeat` ticks later:

```json
{"type":"step",
 "obs":["...32 rows of obs_size..."],
 "reward":["...32 floats..."],
 "terminated":["...32 booleans..."],
 "truncated":["...32 booleans..."],
 "outcome":["...32 ints..."],
 "final_obs":{"7":["...obs_size floats..."],"19":["..."]}}
```

The field names are Gymnasium's, so the Python wrapper passes them straight through.

### `terminated` and `truncated` are two fields, and this is the important part of the file

`terminated` means the episode **really ended**: the ship landed, crashed, ditched, left
the system. There is no future after it, and its value is zero.

`truncated` means the episode **was interrupted**: the clock ran out while the ship was
still flying perfectly well. There is a future, you just stopped watching. Its value is
whatever the critic says the ship was worth at that moment.

Collapse these into one flag and the trainer cannot tell them apart, and the bug is
unfixable downstream no matter how careful the Python is. Treating a time cutoff as a
real ending teaches the agent that the clock running out is as bad as hitting the
ground. Treating a crash as an interruption teaches it the opposite. An earlier version
of this got it backwards and produced an advantage of +45.835 where the correct value
was −4.015. CleanRL's PPO collapses them as shipped; the copy in `python/` does not.

If a ship both ends for real and hits the limit on the same tick, `terminated` wins and
`truncated` is false. Say so once, here, so both sides implement the same rule.

A ship that ends partway through a held action coasts with its engines off until the
decision boundary. That coast is not scored, so a reward never mixes two episodes.

### `final_obs`, and why it has to exist

The engine resets a finished ship at the decision boundary, so the row in `obs` for that
ship is already the **first frame of its next episode**. The last frame of the episode
that just ended would otherwise be lost — and that is exactly the frame the critic has
to look at to carry value across a time cutoff.

So `final_obs` carries it: a map from ship index to that ship's last observation, taken
on the tick the episode ended, holding an entry for exactly those ships whose
`terminated` or `truncated` is true, and nothing else. Without this field, correct
truncation handling is impossible. With it, it is three lines.

## 3. Reset and close

```json
{"type":"reset"}   →  {"type":"reset","obs":["...32 rows..."]}
{"type":"close"}   →  {"type":"close"}, then the engine exits
```

`reset` starts every ship on a new episode and uses no simulated time. `close` is
answered before the engine exits so the trainer knows the request arrived; if the
trainer disconnects without sending it, the engine notices and exits anyway.

## Errors

```json
{"type":"error","reason":"step needs 32 actions"}
```

Sent instead of the normal reply when a request is malformed, arrives out of order
(`reset` or `step` before `hello`), or has an unknown `type`. The trainer raises on it
immediately.

## Rules both sides enforce

1. Every array is exactly `ships` long, every observation row exactly `obs_size`, every
   action row exactly `act_size`.
2. Every float is finite. The engine must never emit `NaN` or infinity — assert it on
   the way out and again on the way in. A `NaN` reaching the optimiser destroys the
   network silently, and the first symptom is a success rate of zero an hour later.
3. `final_obs` holds an entry for every ended ship and no others.
4. Boolean flags are JSON `true`/`false`. In GDScript that means `Array[bool]`:
   `JSON.stringify` writes a `PackedByteArray` as a string, and in Python any non-empty
   string is true.
5. The trainer times out rather than blocking forever. A headless Godot that failed to
   parse a script keeps running and never replies, so a hang means a parse error, not a
   slow step. See the reimport note in `README.md`.

## First test

Before PPO exists: connect, send random actions, print throughput and the ending each
episode got. A random policy should crash almost every time. If it does not, the
termination logic is wrong, and that is worth knowing in a minute rather than an hour
into a training run.
