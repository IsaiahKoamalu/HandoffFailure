# The planet system

Reference for how planets, terrain, oceans, and terrain collision fit together.
Covers `planet.gd`, `planet_mesh.gd`, and the parts of `main.gd` that drive them.

---

## Architecture at a glance

| File | Role |
|---|---|
| `main.gd` (`SolarSystem`) | Owns the simulation clock, drives every planet in parent-first order, answers gravity and sphere-of-influence queries, streams terrain collision to registered bodies. |
| `planet.gd` (`Planet`) | One body. Analytic orbit, gravity, procedural terrain height, ocean, collision patch. |
| `planet_mesh.gd` (`PlanetMesh`) | Pure geometry. Builds cube-sphere meshes and collision patches from a planet's height function. Stateless — all static methods. |

The design rule throughout: **terrain is a pure function of direction and seed.**
Nothing is baked or cached to disk. That is what allows collision to be built on
demand, meshes to be rebuilt at any resolution, and the whole system to be
reproducible across reloads.

---

## Orbits

Orbits are evaluated **analytically**, not integrated. `init()` places each body
on an exactly circular path and precomputes a signed `_mean_motion`; from then
on `update_orbit(time)` is a closed-form function of the simulation clock.

This buys three things a per-tick integrator could not:

- **Time warp.** `SolarSystem.time_scale` fast-forwards the system exactly, not
  approximately. No substepping, no accuracy loss.
- **No drift.** Orbital radius holds exactly. An integrator loses altitude
  slowly — measurably so over a single orbit.
- **Determinism.** Position depends only on `time`, so a reload reproduces the
  same system state.

`SolarSystem` drives `update_orbit()` over its depth-sorted `Planets` array, so
a moon always reads its parent's position for the *current* tick. Planets must
never integrate themselves in their own `_physics_process` — that reintroduces
an ordering bug where a moon lags its parent by one frame.

### Sign convention

`_mean_motion` is **negative**. The orbit runs in the direction
`get_orbital_velocity()` reports, which is clockwise about `Vector3.UP`. Spin
uses the same convention. If you change one, change both.

---

## Terrain

### The height function

`get_terrain_height(dir)` takes a **planet-local, normalized** direction and
returns a radial offset from `surface_radius`.

Local, not world — terrain is fixed to the surface and must rotate with the
planet's spin. Feeding it a world-space direction makes the terrain slide
around underneath the planet as it turns.

Three noise layers, all `FastNoiseLite` sampled in **3D** on the unit sphere:

| Layer | Fractal | Purpose |
|---|---|---|
| `_continents` | FBM, 3 octaves | Low-frequency mask deciding land from basin |
| `_detail` | FBM, 5 octaves | Rolling relief on land |
| `_mountains` | Ridged, 4 octaves | Sharp peaks, squared so foothills stay low |

Sampling in 3D rather than over UVs is deliberate: it removes both the seam
where `u` wraps and the pinching at the poles, because there is no
parameterization to distort. Frequency reads as **features per planet** — around
`1.0` gives a handful of continents.

The detail layers are multiplied by the continent mask. Without that, the whole
planet is uniformly bumpy instead of having distinct continents and seabeds.

### Terrain character

`detail_weight` and `ridge_weight` scale the two land layers independently, and
they are what make one planet look unlike another:

| | `detail_weight` | `ridge_weight` | reads as |
|---|---|---|---|
| rolling | 0.45–0.50 | 0.40–0.55 | hills, broad valleys |
| balanced | 0.35 | 0.85 | default |
| alpine | 0.25 | 1.10 | sharp ranges, narrow passes |

Combined with `continent_frequency` (few large landmasses vs many small ones)
and `sea_level_bias`, that is enough to make a temperate world, a dry mountain
world and an archipelago from the same function.

### Normals

`PlanetMesh` computes per-vertex normals from the height field's own gradient
rather than calling `SurfaceTool.generate_normals()`, which averages face
normals and leaves the coarse mesh visibly faceted.

Each face samples **one ring wider than the output grid**. Those extra samples
exist only to be neighbours for the edge vertices' central differences, which is
what keeps normals continuous across the seam between cube faces. At resolution
160 the ring costs about 2.5% more work — far cheaper than the extra height
evaluations a per-vertex finite difference would need.

Measured across all three planets: 910,116 normals each, none non-unit, none
facing inward, and a mean tilt from radial of 10–22° — they follow terrain
slope rather than just the sphere.

### Colour

`get_terrain_color()` returns a per-vertex colour from a six-stop hypsometric
ramp — `color_deep` through `color_peak` — and `PlanetMesh` writes it into
`ARRAY_COLOR`. The material sets `vertex_color_use_as_albedo`.

**This is not optional decoration.** `PlanetMesh.build()` returns a bare
`ArrayMesh` with no material of its own, so without it every planet renders in
Godot's default white — no shading cue, no way to read terrain at a glance.

The land ramp is scaled against `get_max_terrain_height() * 0.66`, not the full
ceiling. Measured peaks land near two thirds of the theoretical maximum because
the layers rarely max out together, so scaling to the ceiling means the snow
line never appears.

### Sea level and basins

`surface_radius` **is** sea level. Where the continent mask goes negative,
`get_terrain_height` returns a *negative* value, shelving smoothly down to
`get_min_terrain_height()`.

`get_min_terrain_height()` is exact, not an estimate — verified to match the
observed minimum across amplitude and depth settings. Size things against it
rather than guessing a margin.

---

## The ocean

A smooth sphere at `surface_radius`, built by `PlanetMesh.build(null, res, radius)`
in fixed-radius mode.

**The ocean has no collision shape.** Because it is a perfect sphere of known
radius, submersion is one comparison:

```gdscript
func is_underwater(world_pos: Vector3) -> bool:
	return global_position.distance_squared_to(world_pos) < surface_radius * surface_radius
```

That is cheaper than a physics query, has no signal-ordering hazards, and — the
real reason — lets the ship fly *into* the sea instead of bouncing off it.
`get_depth_at()` gives a signed depth suitable for driving buoyancy strength or
an underwater screen tint.

The material sets `cull_mode = CULL_DISABLED`. With default back-face culling
the ocean surface vanishes the moment the camera drops below it.

### Buoyancy and swimming

Both the ship and the astronaut consume `is_underwater()` / `get_depth_at()`.
They solve the same problem in deliberately different ways, because they live in
different frames.

**The astronaut** (`_process_swimming`) replaces free fall with buoyancy plus
drag. Buoyancy is applied as `_gravity * (1.0 - _buoyancy)`, so a value above
`1.0` drifts toward the surface. Drag lerps `velocity` toward the swim input,
which both stops the astronaut coasting forever and caps the buoyant rise at a
terminal speed. Swim input deliberately is *not* flattened onto the tangent
plane the way walking is — underwater you go where you are looking, and
`move_up` / `move_down` work.

Note that the astronaut needs no special handling for the water moving with the
planet: it is reparented into the planet's frame, so its `velocity` is already
frame-relative and lerping toward zero means "stationary relative to the water".

**The ship** (`_process_buoyancy`) is a `RigidBody3D` in world space, and gets
neither of those for free. Two things it must do that the astronaut does not:

1. **Ramp buoyancy by submerged fraction**, over `_hull_depth`. A binary
   underwater test has no equilibrium — full buoyancy throws the ship clear of
   the sea, gravity drops it back, and it never settles. Ramped, it comes to
   rest where buoyancy cancels gravity, which is a float. Equilibrium depth is
   `_hull_depth / _buoyancy`; at the defaults that is 1.74 units, and measured
   1.744.
2. **Apply drag relative to the water, not the world.** Godot's `linear_damp`
   damps toward world-stationary, but the ocean rides the planet's orbit and
   spin. Using it drags the ship sideways at orbital speed — measured as a
   tangential drift climbing to ~10 units/s on a planet orbiting at 10. The fix
   is to subtract `get_surface_velocity()` and damp the remainder by hand. This
   is the same class of bug as the moving-platform double-count the astronaut
   avoids by being reparented.

Drag is *not* ramped by the submerged fraction, only buoyancy is. Fading drag
out near the surface removes the only thing settling the ship into equilibrium.

`_process_parking` is gated on `not is_submerged`. Parking pins the ship to a
fixed surface-local point, so buoyancy would accumulate velocity against a
frozen position and release it all at once on un-parking.

---

## Collision

Three layers, in order of precedence:

### 1. Streamed terrain patches (`TerrainPatch_<client>`)

A `ConcavePolygonShape3D` covering a square patch centred on a client, rebuilt
once that client drifts a third of the way to the patch edge.

A full-sphere collision mesh at walkable detail is not an option — at radius 500
that is roughly 6M triangles. A patch at `patch_resolution = 48` is 4,418.

`patch_extent` is a **distance in world units** — you need a fixed amount of
ground underfoot regardless of planet size — capped by `patch_max_angle` so that
on a small moon it cannot wrap most of the way round and break the tangent-plane
construction. `_patch_angle()` returns `min(patch_max_angle, patch_extent / R)`.

An angle alone does not work: at radius 20 a 30-unit patch would span 1.5 rad,
while at radius 260 a 0.35 rad patch would span 91 units and rebuild far more
mesh than anything needs.

Patches are built in **planet-local space**, so the `CollisionShape3D` holding
one sits at the planet's origin with an identity transform.

#### One patch per client, not one per planet

`Planet._patches` is keyed by client node. This is not an optimisation — it is
required. With a single shared patch, two clients standing apart on the same
planet (a parked ship and an astronaut who has walked off) each recentre it
every frame, so both spend alternate frames with no ground beneath them and fall
through the world. Measured as 1,108 rebuilds across 1,170 frames before the
split, versus the expected 42.

`SolarSystem` calls `release_client()` when a client's dominant planet changes,
so colliders do not accumulate on every body the ship has visited.

#### Builds run on a worker thread

`PlanetMesh.build_patch` is a pure function of the planet's seed and touches no
scene state, so `update_collision_for` hands it to `WorkerThreadPool` and adopts
the result on a later frame. The old shape stays live until the new one lands,
so there is never a gap in collision.

Concurrent `FastNoiseLite` reads are safe — verified by building 64 patches
across all threads against one shared noise set and comparing byte-for-byte with
serial output: zero mismatches, zero vertex error, over repeated runs.

`build_collision_now()` is the synchronous escape hatch, used by
`Astronaut.place_at()` because a fresh spawn cannot wait a frame for ground.

Two rules worth keeping:

- **Always call `wait_for_task_completion`**, even after `is_task_completed`
  returns true — it is what releases the task's resources. `_exit_tree` waits on
  every in-flight task so a worker never outlives its planet.
- **The `is_task_completed` check is the memory barrier** for reading
  `Patch.result`. Do not read it anywhere else.

### 2. Sea-level backstop (the authored `SphereShape3D`)

Deliberately kept, shrunk to just inside the deepest possible seabed:

```gdscript
sphere.radius = (surface_radius + get_min_terrain_height()) * 0.995
```

Because it sits below all terrain, it can never poke through the surface. It
catches anything reaching the planet before a patch has streamed in — a ship
approaching at speed, or a body that was never registered as a terrain client.

The scene **shares one `SphereShape3D` resource across every planet**, so
`_build_terrain` duplicates it before resizing. Resizing in place would resize
all of them.

### 3. Terrain clients

`SolarSystem.register_terrain_client(node)` opts a body into collision
streaming. The astronaut and the ship both register.

The ship matters as much as the astronaut: its `_ground_cast` and
`_ramp_lip_on_ground()` raycast against terrain and **fail silently** against
ground that was never generated.

Registration is deferred by one frame — `SolarSystem.Instance` is assigned in
Main's `_ready`, which Godot runs *after* its children's.

---

## Frames of reference

The astronaut is reparented to whatever it is riding — a planet, the ship, or
the solar system in open space — and its `velocity` is **frame-relative**.
Standing still on a planet means `velocity == Vector3.ZERO` despite the planet
orbiting at speed.

Because of this the astronaut scene sets `platform_floor_layers = 0`. A planet
is a `StaticBody3D` carrying `constant_linear_velocity`, so Godot's
moving-platform handling would add the orbital velocity a second time on top of
what reparenting already provides.

Gravity stays perfectly radial and is computed from `surface_radius`, ignoring
terrain. That is correct — real planets have no gravity anomaly at mountain
scale — and it keeps the astronaut's "up" vector well-defined on slopes, which
`floor_max_angle` then handles.

---

## Tuning reference

| Parameter | Default | Notes |
|---|---|---|
| `terrain_amplitude` | `0.06` | Peak height as a fraction of radius. `0.05` is Earth-like. Past `0.15` it stops reading as a planet. |
| `sea_level_bias` | `0.15` | Raise to flood, lower to dry out. `0.15` gives ~31% ocean coverage. |
| `ocean_depth` | `0.35` | Basin depth as a fraction of `terrain_amplitude`. |
| `mesh_resolution` | `64` | Triangles = `6*(n-1)^2*2`. 64 → 47k, 128 → 190k, 160 → 303k. |
| `far_resolution` | `64` | Distant stand-in mesh, ~48k triangles. |
| `detail_range_factor` | `5.0` | Planet radii at which the stand-in takes over. |
| `detail_weight` | `0.35` | Rolling fBm weight. |
| `ridge_weight` | `0.85` | Ridged weight. Raise for alpine. |
| `patch_extent` | `30.0` | Collision patch half-width, world units. |
| `patch_max_angle` | `0.35` | Radians. Caps the above on small bodies. |
| `water_resolution` | `32` | 11.5k triangles. Can stay low until waves are added. |
| `patch_half_angle` | `0.35` | Radians. ~20° of arc. |
| `patch_resolution` | `48` | 4,418 triangles per patch. |
| `continent_frequency` | `1.1` | Features per planet. |
| `time_scale` | `1.0` | On `SolarSystem`. Exact at any value. |

### Water handling

| Parameter | Where | Default | Notes |
|---|---|---|---|
| `_buoyancy` | Astronaut | `1.08` | Multiple of local gravity. Above `1.0` rises. |
| `_swim_speed` | Astronaut | `3.0` | Versus `_walk_speed` of `5.0`. |
| `_water_drag` | Astronaut | `2.5` | Sets terminal rise rate against buoyancy (~0.35). |
| `_buoyancy` | Ship | `1.15` | Above `1.0` surfaces, below settles on the seabed. |
| `_water_drag` | Ship | `3.0` | Applied to water-relative velocity, not `linear_damp`. |
| `_hull_depth` | Ship | `2.0` | Ramp distance. Float depth is `_hull_depth / _buoyancy`. |

### Vertex spacing

A cube face spans 90° of arc, so spacing is `(π/2)·R / (resolution − 1)`:

| Radius | `mesh_resolution` | Spacing | Triangles |
|---|---|---|---|
| 20 | 64 | 0.50 | 47k |
| 200 | 64 | 5.0 | 47k |
| 200 | 128 | 2.5 | 190k |
| 500 | 128 | 6.2 | 190k |

---

## Invariants worth not breaking

- **`get_terrain_height` takes a planet-LOCAL direction.** World-space input
  makes terrain slide as the planet spins.
- **The backstop must sit below `get_min_terrain_height()`.** Size it against
  `surface_radius` and you get an invisible sphere floating above every seabed.
- **Winding is clockwise-front.** Godot treats clockwise as front-facing, the
  opposite of the right-hand rule. Writing triangles in the order that "looks
  right" turns the planet inside out.
- **No `generate_tangents()`.** The cube-sphere carries no UVs; it errors. The
  triplanar material does not need them.
- **Spheres of influence must not overlap.** `CAPTURE_RADIUS_FACTOR` is
  `2.0 × surface_radius`. Two planets whose capture radii overlap make
  `get_dominant_planet` return the wrong body, handing the astronaut to a planet
  they are not standing on.
- **Both patch and initial mesh generation are threaded.** `SolarSystem._ready`
  builds every planet's mesh concurrently via `add_group_task`, then attaches
  them on the main thread — scene mutation is not thread-safe. At resolution 160
  that took three planets from 1.65 s serial to ~0.86 s.
- **Anything in world space must handle water moving with the planet.** The
  astronaut gets this free from reparenting; the ship does not. Reaching for
  `linear_damp` on an orbiting body is the trap.
- **Oceans must be deep enough to swim in.** With `terrain_amplitude = 0.08` and
  `ocean_depth = 0.35` on a radius-20 planet, the deepest water is 0.56 units
  against a ~1.9-unit astronaut. It is a puddle: you stand on the seabed with
  your head well clear. Depth scales with `terrain_amplitude * ocean_depth *
  surface_radius`, so small planets need a proportionally larger amplitude.

---

## Distance culling

Each planet carries two meshes: the detail mesh at `mesh_resolution`, and a
stand-in at `far_resolution` (64 by default, ~48k triangles). The detail mesh
has `visibility_range_end = surface_radius * detail_range_factor`; the stand-in
has a matching `visibility_range_begin`, so the ranges meet exactly and there is
never a distance where neither draws.

The default factor of 5 switches at about 23° of angular size. Without this,
every planet drew its full mesh at all times — a body 59 pixels tall on screen
cost the same 303k triangles as the one under your feet, about 87 triangles per
pixel of its screen area.

From the spawn point that takes the standing cost from 910,116 triangles to
398,628, a 56% saving, for 136 ms more at load (860 → 996 ms).

`visibility_range_fade_mode` is deliberately `DISABLED`, giving a hard switch.
`FADE_SELF` would cross-fade instead, but Godot applies the fade margin to one
side of the threshold and choosing the wrong side leaves a band where neither
mesh draws. It is worth trying with the viewport open; it is not worth guessing.

## Navigation

`SolarSystem.target` plus `cycle_target()` on **[Tab]**, and
`get_target_vector()` returning range-to-surface and closing speed. The HUD
renders where you are (dominant body and altitude above terrain), where you are
going, and an ETA.

Closing speed subtracts the target's own `constant_linear_velocity` — the
destination is orbiting, so a raw dot product against your world velocity would
report closing on a planet you are actually falling behind.

The astronaut's `velocity` is frame-relative, so `get_world_velocity()` adds the
frame's motion back before any navigation maths.

## Save and load

**[F5]** / **[F9]**, to `user://quicksave.dat` via `store_var`.

The file is tiny, and that is a direct payoff from two earlier decisions:
orbits are analytic, so the entire solar system is recoverable from a single
float; terrain is a pure function of a seed, so none of it is serialised at all.
What remains is where the player left their ship and their body.

Load winds `time` back and re-runs `update_orbit()` on every planet **before**
placing anything, so bodies restored relative to a planet land against that
planet's restored position. `Astronaut.restore()` takes a world velocity and
converts it into the frame it turns out to be riding.

Verified as an exact round trip: time 6.67 → 1.00, ship position, target, ramp
and both light states all restored, and every planet back where it was.

## Known limitations

- **No near-field LOD.** One fixed-resolution mesh per planet up close, 303k
  triangles, 1.4–2.6 units between vertices at radius 140–260. There *is*
  distance culling (see below), but nothing raises detail underfoot.

  The gap between the drawn surface and the surface collision follows was
  measured rather than assumed: **mean 0.018–0.024 units, worst case 0.568**,
  against an astronaut 1.80 tall. So this is a visual-quality question, not a
  correctness one — the ground you walk on is where it looks, to within about a
  centimetre. Analytic normals address most of the visual complaint. A second
  high-detail tier is still the right answer past radius ~300, but it needs
  visual iteration to get the seam and depth handling right.
- **Adopting a finished patch still costs the main thread.** Handing the shape
  to the physics server measures ~1.8 ms, on the ~42 frames out of 1,170 where a
  rebuild lands. The generation itself is fully off-thread.
- **Float precision caps world size.** Float32 gives ~7 significant digits, so
  beyond roughly 1e6 units from the origin the astronaut visibly jitters. Past
  that you need a floating origin.
- **Ocean is a flat sphere.** No waves, no depth-based colour, no refraction.
- **Nothing to find.** The terrain supports exploration; there is no content in
  it. Because height is a pure function of a seed, points of interest can be
  placed deterministically with no authoring and no storage — pick a direction,
  query `get_surface_radius_at`, drop something there.
- **Sky is the same everywhere.** The Milky Way panorama is the background on
  every surface. Per-planet atmosphere would do more for "these are different
  worlds" than further terrain tuning.
- **No underwater screen effect.** The HUD flags submersion in text only; there
  is no tint, fog, or muffling.
- **Buoyancy is uniform, not per-volume.** The ship floats level rather than
  settling by its actual displaced shape, and does not right itself from an
  upside-down entry any faster than the existing auto-orientation manages.
- **Surface bobbing is untuned.** Both bodies settle correctly, but neither has
  wave interaction to bob against once waves exist.

---

## The current system

| body | radius | orbit | character | walk lap |
|---|---|---|---|---|
| Sun | 200 | — | star, no terrain | — |
| Planet1 | 180 | 1200 | temperate, rolling, real oceans | 226 s |
| Planet2 | 260 | 2400 | dry alpine, sharp ranges, little water | 327 s |
| Planet3 | 140 | 3600 | archipelago, many islands, deep water | 176 s |

Sphere-of-influence separations all clear their combined capture radii, the
tightest being Planet1↔Planet2 at 1200 against a required 880.

Two things that scaling breaks and are easy to forget:

- **The sun's `omni_range`** was 1000, which no longer reached past the first
  orbit. It is now 6000. `omni_attenuation` stays at 0.061.

  Be careful with that value. In Godot 4 `omni_attenuation` is a **decay
  exponent applied as `distance^(-decay)`**, not a shaping exponent on a
  normalised 0–1 falloff. Raising it to 1.5 makes the term `1200^-1.5 ≈ 2.4e-5`
  at the first orbit, which turns every planet black. Any increase has to come
  with a very large increase in `light_energy` to match — and the only way to
  judge the result is to look at it.
- **Camera `far`** defaults to 4000. With the outer planet at 3600 plus the
  ship's own distance from origin, distant bodies clipped out entirely. Both
  cameras are now `near = 0.1`, `far = 20000`.

## Verification

Behaviour confirmed on Godot 4.7.2, headless, against `main.tscn`:

```
frame  60  alt=20.711 ground=19.451 above=+1.260 on_floor=false underwater=false
frame 200  alt=20.414 ground=19.451 above=+0.963 on_floor=true  underwater=false

  Sun       ocean_tris=    0 backstop=50.000
  Planet1   ocean_tris=11532 backstop=19.343
  Planet2   ocean_tris=11532 backstop=39.303
  Planet3   ocean_tris=11532 backstop=19.343
```

The astronaut falls onto terrain and comes to rest at `+0.963` — the capsule's
half-height — and stays there. Note `ground = 19.451`: that drop site is a
shallow basin *below* sea level, with the astronaut's origin still above water.

Mesh integrity, checked at resolution 48 on a radius-200 planet: 26,508
triangles, all 79,524 vertex normals pointing outward, collision patches
agreeing with the render surface to 4×10⁻⁶.

Patch streaming, sweeping a client around Planet3 at walking pace for 1,170
frames:

```
patch rebuilds       = 42  (expected ~42)
frames with NO shape = 1   (the first, before the initial build lands)
main-thread cost     = 0.0673 ms avg, 2.3830 ms worst
```

Against 3.36 ms **every** rebuild when the build was synchronous. The clean
project runs with zero bytes on stderr.

Water handling, dropped into Planet2's ocean (seabed 39.50, sea level 50.0,
both entering at 44.75):

```
f 400 | SHIP alt=48.115 depth=+1.88 drift=0.459 | ASTRO alt=46.887 depth=+3.11 |v|=0.344
f 900 | SHIP alt=48.256 depth=+1.74 drift=0.014 | ASTRO alt=49.594 depth=+0.41 |v|=0.307
f1400 | SHIP alt=48.257 depth=+1.74 drift=0.013 | ASTRO alt=49.997 depth=+0.00 |v|=0.030
```

The ship settles at 1.74 below sea level — matching the predicted
`_hull_depth / _buoyancy` — with water-relative drift decaying to 0.013 and
world velocity tracking the planet's 10.000 units/s orbit. The astronaut rises
at ~0.31 terminal and floats at exactly sea level.
