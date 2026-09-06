"""Score a controller on the landing task.

Runs a fixed number of episodes and reports landing rate, touchdown speed and
return. Two controllers are built in:

``--controller baseline``
    A hand-written proportional descent controller: hold a descent rate that
    tapers with altitude, and torque the belly toward the ground. This is the
    thing you would write *instead* of training something, and it is the number
    the policy has to beat for any of this to have been worth doing.

``--controller policy``
    The trained network, evaluated at its distribution mean rather than sampled.
    Exploration noise has no job left once training is over and it shows up as
    visible twitching.

``--controller zero``
    Hands off the controls. Establishes what the ship's own flight assist does
    unaided, which is the real floor — not a random policy.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from godot_env import ACT_SIZE, RUNNING, LANDED, GodotVecEnv

# Anchored to this file rather than the working directory. Every documented
# invocation is from python/, so a bare relative path quietly resolved to a
# stray python/rl/ instead of the real run directory -- and loaded an old
# policy rather than reporting that it could not find the new one.
DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "rl/runs/landing/policy.json"

# Fuel fraction is the last slot of the observation; see rl/landing_env.gd.
FUEL_SLOT = 19

# Observation layout, mirroring LandingEnv.build_observation.
ALT = 0
V_LOCAL = slice(2, 5)
UP_LOCAL = slice(5, 8)
V_UP = 8
LANDNESS = 14
TO_SITE = slice(15, 18)
SITE_ANGLE = 18

# Do not commit to a descent right on the waterline. The land signal is a
# smooth field and the shoreline is where it crosses zero, so a touchdown aimed
# at exactly zero lands half the time in the surf.
SHORE_MARGIN = 0.05
# Altitude, in radii, held while crossing to the target continent. High enough
# to clear the terrain by a wide margin and low enough that arriving does not
# then need a long descent.
CRUISE_ALT = 0.16
# Angle to the centre of the continent, in radians, inside which dry ground can
# be assumed to belong to it. The target covers 12-30% of a planet, so its
# angular radius runs to about a radian; this is comfortably inside that. Being
# generous here matters -- insisting on the exact centre meant flying past
# perfectly good landing ground to reach a point, and then hovering over it.
ARRIVED_ANGLE = 0.55
# Floor on the closing rate while over water. Without it a ship that arrives
# above a bay stalls: it is close enough that the speed target has decayed to
# nothing, and wet enough that it refuses to descend, so it hovers until the
# clock runs out. This keeps it walking until there is ground beneath it.
WET_CLOSING_FLOOR = 0.4
# Proportional gain on the horizontal velocity error.
BRAKE_GAIN = 8.0
# Horizontal speed to hold while crossing, in observation units (~42 units/s per
# unit), so about twelve units a second. The ceiling is what is left of the
# thrust budget after holding altitude: ask for much more and the vector tips so
# far off vertical that the ship sinks while it travels.
CRUISE_SPEED = 0.30


# --- Docking -----------------------------------------------------------------
# Observation layout, mirroring DockingEnv.build_observation. A docking
# observation is a different vector of a different width; nothing here is
# shared with the landing slots above, and the names are kept apart so that a
# mistake reads as one.
D_RANGE = 0
D_OFFSET = slice(2, 5)
D_VELOCITY = slice(5, 8)
D_CLOSING = 8
D_NORMAL = slice(9, 12)
D_ALIGN = 12
D_AXIAL = 13
D_LATERAL = 14
D_SPIN = slice(15, 18)
D_UP = slice(22, 25)

# Speed to close at, per unit of distance to the point being flown to, in
# observation units (SPEED_SCALE = 20 units/s per unit, approach_range = 60
# units per unit of distance). Tapers to nothing on arrival, which is what makes
# contact gentle without a separate braking phase.
#
# It has to be high. Flight assist trims drift against the *planet* the station
# orbits, not against the station, so the ship sits in a persistent relative
# drift of about a unit a second. A proportional law against a constant
# disturbance settles at an error of disturbance/gain: at 0.9 that was a
# three-unit standing offset the ship could not close, and every approach ran
# out the clock holding it.
DOCK_CLOSE_GAIN = 9.0
# Ceiling on that, so a long approach does not begin at a sprint.
DOCK_CLOSE_MAX = 0.25
# A second ceiling, tapering with range to the *port*, so contact is gentle.
#
# Two limits rather than one because they are pulling opposite ways. The gain
# above has to be high to beat the orbital drift, and at that gain the ship was
# still commanding its full five units a second two metres from the port and
# arriving at 4.66 against a 1.0 gate. This one does not care how far the
# current aim point is, only how close the port is: 0.75 gives half a unit a
# second at two units out, and meets DOCK_CLOSE_MAX at twenty.
DOCK_ARRIVE_GAIN = 0.75
# Closing rate held once lined up and committed, in observation units: 0.6 units
# a second, against a 1.0-unit gate at contact.
#
# Everything above it is proportional, and a proportional law against a standing
# disturbance stops where its command equals the disturbance. Measured, it
# stopped at a range of 2.4 units -- alignment -1.00, lateral 0.41, closing 0.41
# -- and held that four tenths short of a 2.0-unit gate for the rest of the
# episode. A constant creep on the last few metres is how real docking systems
# close it, and it needs no integral term to wind up.
DOCK_CREEP = 0.03
# Alignment required before committing to that creep. Approaching blind at a
# fixed rate while still crooked is how a botch becomes a collision.
DOCK_COMMIT_ALIGN = -0.95
# Gain on velocity error. The ship is a rigid body with flight assist trimming
# against a planet that is not the station, so this has real work to do.
DOCK_VELOCITY_GAIN = 6.0
# Gain turning the pointing error into a turn rate, and the one damping spin.
DOCK_TURN_GAIN = 2.5
DOCK_SPIN_GAIN = 1.2
# Gain rolling the ship's up onto the station's. Ports are keyed, and the
# astronaut walks through the join -- a ship latched upside down relative to the
# deck would have gravity reverse in the doorway.
DOCK_ROLL_GAIN = 2.0
# Lateral offset, in approach ranges, at which closing is throttled to nothing.
# 0.025 of a 60-unit range is 1.5 units, against a 1.0-unit gate at contact.
#
# The lateral error cannot be driven to zero. Flight assist trims drift against
# the planet the station orbits rather than against the station, so there is a
# standing disturbance, and a proportional law settles where its command
# cancels it -- around 0.4 units, measured.
#
# So this throttles *closing* rather than choosing between closing and
# centring. An earlier version blended the aim point between the port and a
# standoff, and the two commands competed: the lateral error then settled at
# whatever threshold the blend used -- 0.41 units for one value of it, 0.76 for
# another -- and the ship parked there holding a perfect corridor. Correcting
# lateral at full authority always, and throttling only the approach, has no
# such fixed point to sit in.
DOCK_TIGHT = 0.025
# Which end of the ship the port is on, as the sign of `Ship.port_axis.z`: -1
# for the nose, +1 for the stern. It is the stern, because the stern is where
# the ramp is and the ramp is the only door in the hull. Mirrored here rather
# than observed because the observation is already expressed relative to the
# port, so nothing in it says which way the *hull* has to point; get this wrong
# and the controller aims the ship perfectly backwards, which measures as a
# clean 0.0% docked rate rather than as anything that looks like a bug.
SHIP_PORT_FACING = 1.0


# Below this, an observation channel counts as constant and is left at unit
# scale instead of being divided by its own standard deviation.
#
# Not an epsilon guarding a divide by zero -- it is a decision about what counts
# as constant. A channel whose standard deviation over a whole training run is a
# thousandth of its own range never varied, and dividing by it turns a rounding
# difference into an enormous input. Measured: the shipped landing policy has
# obs_std[19] = 1.0e-4 on the fuel channel, having trained with unlimited tanks
# where the value was always exactly 1.0. Giving the player's ship a fuel budget
# then fed it 0.985, which divided out to -147 against inputs that otherwise
# live in [-3, 3], and the ship stopped being able to land.
#
# Lives here rather than in ppo because ppo imports from this module and the
# other way round is a circular import. NeuralPilot carries the same number and
# verify_pilot compares the two paths.
CONSTANT_STD = 1e-3


def dock_baseline_controller(obs: np.ndarray) -> np.ndarray:
    """Fly to the corridor, line up on it, then close.

    A proportional law, and deliberately so -- transit in ShipAutopilot is the
    same shape for the same reason. What makes it more than "thrust at the
    port" is the order of business: a ship that closes on the port before it is
    on the centreline arrives at the hull, so while the lateral offset is large
    this cancels it *first* and only then allows range to come down.

    Everything is in the ship's own frame, which is the frame the commands are
    in, so a direction here maps onto a thrust without any rotation maths.
    """
    n = obs.shape[0]
    action = np.zeros((n, ACT_SIZE), dtype=np.float32)

    offset = obs[:, D_OFFSET]           # ship port -> station port
    velocity = obs[:, D_VELOCITY]       # relative to the station
    normal = obs[:, D_NORMAL]           # outward normal of the port face
    lateral = obs[:, D_LATERAL]
    axial = obs[:, D_AXIAL]
    ranges = obs[:, D_RANGE]

    # The two jobs are commanded separately, and only one of them is throttled.
    #
    # Across the corridor: always corrected, at full gain. `across` is the part
    # of the offset perpendicular to the port normal.
    axial_component = np.sum(offset * normal, axis=1, keepdims=True) * normal
    across = offset - axial_component
    across_distance = np.linalg.norm(across, axis=1, keepdims=True)
    across_speed = np.minimum(across_distance * DOCK_CLOSE_GAIN, DOCK_CLOSE_MAX)
    wanted = np.where(
        across_distance > 1e-6,
        across / np.maximum(across_distance, 1e-6) * across_speed,
        0.0,
    )

    # Along it: gated on being centred, facing the right way, and in front of
    # the port face. A ship failing any of those closes at nothing and spends
    # the time fixing it instead -- which is also what keeps a spawn behind the
    # station out of the hull.
    gate = np.clip(1.0 - lateral / DOCK_TIGHT, 0.0, 1.0)
    gate = gate * (obs[:, D_ALIGN] < DOCK_COMMIT_ALIGN)
    gate = gate * (axial > 0.0)
    approach = np.minimum(ranges * DOCK_ARRIVE_GAIN, DOCK_CLOSE_MAX)
    # A floor, because everything above is proportional and a proportional law
    # against a standing disturbance stops where its command equals it.
    # Measured without it: parked at a range of 2.4 units with alignment -1.00,
    # holding four tenths short of a 2.0-unit gate for the rest of the episode.
    # A constant creep over the last few metres is how real docking systems
    # close that, and it cannot wind up the way an integral term would.
    approach = np.maximum(approach, DOCK_CREEP) * gate
    wanted = wanted - normal * approach[:, None]

    # Behind the port face there is nothing to line up on yet: get out in front
    # of it first.
    wanted = wanted + normal * ((axial <= 0.0) * DOCK_CLOSE_MAX)[:, None]

    action[:, 0:3] = np.clip((wanted - velocity) * DOCK_VELOCITY_GAIN, -1.0, 1.0)

    # Aim the hull. The turn law below drives the ship's nose, local -Z. The
    # port is on the stern, so the nose has to end up pointing back *up* the
    # corridor, away from the station, and the ship backs in.
    d = normal * SHIP_PORT_FACING
    spin = obs[:, D_SPIN]
    action[:, 3] = np.clip(d[:, 1] * DOCK_TURN_GAIN - spin[:, 0] * DOCK_SPIN_GAIN, -1.0, 1.0)
    action[:, 4] = np.clip(-d[:, 0] * DOCK_TURN_GAIN - spin[:, 1] * DOCK_SPIN_GAIN, -1.0, 1.0)
    # Roll is a separate axis and a separate job: bring the ship's own up onto
    # the station's. Rolling about local +Z carries +Y toward -X, so a station
    # up leaning toward +X wants a negative command.
    up = obs[:, D_UP]
    action[:, 5] = np.clip(
        -up[:, 0] * DOCK_ROLL_GAIN - spin[:, 2] * DOCK_SPIN_GAIN, -1.0, 1.0)
    return action


def baseline_controller(obs: np.ndarray) -> np.ndarray:
    """Fly to the planet's largest continent from anywhere, and land on it.

    Everything is in the ship's own frame, which is what keeps it short:
    `up_local` is where the ground is from the ship's point of view, so
    thrusting along it is thrusting away from the ground, and rotating it onto
    +Y is levelling out.

    Two phases, chosen by how far the target still is:

    **Cruise.** Hold a set altitude, kill sideways drift, and push along the
    bearing to the continent. Holding altitude rather than descending is the
    whole trick -- a ship that descends while it travels arrives low, slow and
    over whatever happens to be underneath it, which is usually sea.

    **Descend.** Once over the continent, hold a descent rate that tapers to a
    crawl, keep the belly down, and refuse to go below a working altitude while
    the ground underneath is still wet, so a coastal approach does not finish in
    the surf.

    The bearing is a great-circle heading to a fixed point, so this works from
    anywhere on the sphere including the far side -- which the previous
    gradient-following version could not do. That one walked to the *nearest*
    coast and stalled in open ocean where the local gradient vanishes.
    """
    action = np.zeros((obs.shape[0], ACT_SIZE), dtype=np.float32)

    altitude = obs[:, ALT]
    v_local = obs[:, V_LOCAL]
    up_local = obs[:, UP_LOCAL]
    v_up = obs[:, V_UP]
    landness = obs[:, LANDNESS]
    to_site = obs[:, TO_SITE]
    site_angle = obs[:, SITE_ANGLE] * np.pi

    # Readiness to descend, as a continuous quantity rather than a switch.
    #
    # Descending needs dry ground below *and* a target close enough that the
    # ground can be assumed to belong to it. Either alone is not enough: dry
    # ground far away is the wrong continent, and being close with water
    # underneath is a bay.
    #
    # Written as a hard boolean this controller works fine and clones badly. A
    # network fitted to it by least squares cannot reproduce a step, so it
    # learns a smeared version and then half-commits -- descending over water
    # sometimes, hovering over good ground other times. Making the expert itself
    # smooth gives the regression something it can actually represent, and costs
    # the controller nothing.
    ready = (
        np.clip((landness - SHORE_MARGIN) / 0.12, 0.0, 1.0)
        * np.clip((ARRIVED_ANGLE - site_angle) / 0.25, 0.0, 1.0)
    )
    cruising = 1.0 - ready
    wet = landness < SHORE_MARGIN

    # --- vertical ---------------------------------------------------------
    # Cruising holds CRUISE_ALT. Descending tapers toward the surface. The
    # observation is scaled by orbital speed (~42 units/s), hence small numbers.
    approach = -0.012 - 0.13 * np.clip(altitude / 0.20, 0.0, 1.0)
    hold = np.clip(1.2 * (CRUISE_ALT - altitude), -0.10, 0.10)
    desired = ready * approach + cruising * hold
    # Gain high enough to saturate on the errors that matter. At 15 a three
    # unit a second overshoot asked for only half thrust, which under the old
    # thrust-to-weight ratio was academic and afterwards was just slack.
    lift = np.clip(45.0 * (desired - v_up), -1.0, 1.0)

    # --- horizontal -------------------------------------------------------
    # A velocity controller, not a brake plus a shove. Those are the same thing
    # when the wanted velocity is zero, and they fight each other when it is
    # not: a braking term proportional to all horizontal velocity cancels the
    # very speed the cruise is trying to build, and the ship sits still burning
    # the clock. Asking for a *target* velocity makes both cases one expression.
    v_horizontal = v_local - up_local * np.sum(v_local * up_local, axis=1)[:, None]

    # Ease the target off as the continent is reached, so the ship arrives
    # stopped rather than sailing past and having to come back.
    closing = np.clip(site_angle / 0.30, 0.0, 1.0)
    closing = np.where(wet, np.maximum(closing, WET_CLOSING_FLOOR), closing)
    v_wanted = to_site * (CRUISE_SPEED * closing * cruising)[:, None]
    lateral = np.clip(BRAKE_GAIN * (v_wanted - v_horizontal), -1.0, 1.0)

    action[:, 0:3] = up_local * lift[:, None] + lateral

    # --- attitude ---------------------------------------------------------
    # Rotate the ship's +Y onto up_local. The axis of that rotation is
    # (0,1,0) x up_local = (up_local.z, 0, -up_local.x), so its X component is
    # pitch and its Z component is roll. The gain is high because a spawn can be
    # tumbling and every tick spent sideways is a tick not descending.
    action[:, 3] = np.clip(4.0 * up_local[:, 2], -1.0, 1.0)
    action[:, 5] = np.clip(-4.0 * up_local[:, 0], -1.0, 1.0)
    return np.clip(action, -1.0, 1.0)


class PolicyController:
    """Forward pass over the exported JSON, in numpy.

    Reads the same file Godot reads, so a disagreement between this number and
    in-game behaviour is a bug in the GDScript forward pass, not in the policy.
    """

    def __init__(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["act_size"] == ACT_SIZE
        # Checked against the engine once it has said which task it is running;
        # this file cannot know that on its own.
        self.obs_size = int(data["obs_size"])
        self.task = data.get("task", "landing")
        self.layers = [
            (np.asarray(layer["w"], dtype=np.float32), np.asarray(layer["b"], dtype=np.float32))
            for layer in data["layers"]
        ]
        self.mean = np.asarray(data["obs_mean"], dtype=np.float32)
        self.std = np.asarray(data["obs_std"], dtype=np.float32)
        self.trained_steps = data.get("trained_steps", 0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # A channel that never moved during training is unit scale, not
        # something to divide by. Same rule and same constant as ppo.Normaliser
        # and NeuralPilot; verify_pilot compares this path against that one and
        # will say so if the three ever disagree -- it did, when only two of
        # them had been changed.
        scale = np.where(self.std < CONSTANT_STD, 1.0, self.std)
        x = np.clip((obs - self.mean) / scale, -10.0, 10.0)
        for index, (weight, bias) in enumerate(self.layers):
            x = x @ weight.T + bias
            if index < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--controller", choices=["policy", "baseline", "zero"], default="policy")
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--task", default="landing")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--port", type=int, default=11040)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-ticks", type=int, default=20000)
    parser.add_argument("--fuel", type=float, default=0.0,
                        help="seconds of full thrust per episode. 0 is unlimited.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="physics ticks before an episode is truncated. 0 keeps "
                             "the env default of 9000.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="physics ticks per decision. Must match what the "
                             "policy was trained at, or the comparison is between "
                             "two different control problems.")
    args = parser.parse_args()

    if args.controller == "policy":
        controller = PolicyController(Path(args.policy_path))
        label = f"policy ({controller.trained_steps:,} steps)"
    elif args.controller == "baseline":
        controller = dock_baseline_controller if args.task == "docking" else baseline_controller
        label = "hand-written baseline"
    else:
        controller = lambda obs: np.zeros((obs.shape[0], ACT_SIZE), dtype=np.float32)
        label = "no input (assist only)"

    outcomes: collections.Counter[int] = collections.Counter()
    impacts: list[float] = []
    landed_impacts: list[float] = []
    returns: list[float] = []
    lengths: list[int] = []

    with GodotVecEnv(
        args.godot, args.project, num_agents=args.agents, port=args.port,
        seed=args.seed, repeat=args.repeat, max_steps=args.max_steps,
        fuel_seconds=args.fuel, task=args.task,
    ) as env:
        if isinstance(controller, PolicyController):
            if controller.task != env.task:
                raise SystemExit(
                    f"policy was trained for task '{controller.task}', "
                    f"engine is running '{env.task}'"
                )
            if controller.obs_size != env.obs_size:
                raise SystemExit(
                    f"policy expects {controller.obs_size} observations, "
                    f"engine sends {env.obs_size}"
                )
        obs = env.reset()
        names = env.outcome_names
        running_return = np.zeros(args.agents)
        running_length = np.zeros(args.agents, dtype=int)
        fuel_used: list[float] = []
        landed_fuel: list[float] = []

        for _ in range(args.max_ticks):
            # Fuel left going in to the step, so an episode that ends on this
            # step is measured before autoreset refills the tanks.
            before = obs[:, FUEL_SLOT].copy()
            obs, reward, terminated, truncated, info = env.step(controller(obs))
            running_return += reward
            running_length += 1
            for i in range(args.agents):
                outcome = int(info["outcome"][i])
                if outcome != RUNNING:
                    outcomes[outcome] += 1
                    impacts.append(float(info["impact"][i]))
                    if outcome == LANDED:
                        landed_impacts.append(float(info["impact"][i]))
                if terminated[i] or truncated[i]:
                    if args.fuel > 0.0:
                        used = (1.0 - float(before[i])) * args.fuel
                        fuel_used.append(used)
                        if outcome == LANDED:
                            landed_fuel.append(used)
                    returns.append(float(running_return[i]))
                    lengths.append(int(running_length[i]))
                    running_return[i] = 0.0
                    running_length[i] = 0
            if len(returns) >= args.episodes:
                break

    total = max(len(returns), 1)
    print(f"\n=== {label} ===")
    print(f"episodes            {len(returns)}")
    # The success outcome is id 1 in every task -- landed, docked -- and the
    # engine named it in the handshake, so this reads correctly under both
    # rather than hardcoding one task's vocabulary.
    success = names.get(1, "success")
    print(f"{success + ' rate':<20}{outcomes[1] / total:.1%}")
    if args.fuel > 0.0 and landed_fuel:
        q = np.percentile(landed_fuel, [50, 90, 99])
        print(f"fuel of a landing   median {q[0]:.1f}s  p90 {q[1]:.1f}s  "
              f"p99 {q[2]:.1f}s  max {max(landed_fuel):.1f}s  of {args.fuel:.0f}s")
    print(f"mean return         {np.mean(returns):.2f}")
    # `lengths` counts decisions, and a decision is `repeat` physics ticks. At
    # repeat 6 a 491-decision episode is 49 seconds of flight, not 8.
    ticks = np.mean(lengths) * args.repeat
    print(f"mean episode length {ticks:.0f} ticks ({ticks / 60:.1f}s)"
          + (f", {np.mean(lengths):.0f} decisions" if args.repeat > 1 else ""))
    if impacts:
        print(f"touchdown speed     mean {np.mean(impacts):.2f}  median {np.median(impacts):.2f}"
              f"  p90 {np.percentile(impacts, 90):.2f} units/s")
    if landed_impacts:
        print(f"  of the landings   mean {np.mean(landed_impacts):.2f} units/s")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {names.get(outcome, outcome):<8} {count / total:6.1%}  ({count})")


if __name__ == "__main__":
    main()
