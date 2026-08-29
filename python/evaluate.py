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

from godot_env import ACT_SIZE, OBS_SIZE, OUTCOME_NAMES, RUNNING, LANDED, GodotVecEnv

# Anchored to this file rather than the working directory. Every documented
# invocation is from python/, so a bare relative path quietly resolved to a
# stray python/rl/ instead of the real run directory -- and loaded an old
# policy rather than reporting that it could not find the new one.
DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "rl/runs/landing/policy.json"

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
        assert data["obs_size"] == OBS_SIZE and data["act_size"] == ACT_SIZE
        self.layers = [
            (np.asarray(layer["w"], dtype=np.float32), np.asarray(layer["b"], dtype=np.float32))
            for layer in data["layers"]
        ]
        self.mean = np.asarray(data["obs_mean"], dtype=np.float32)
        self.std = np.asarray(data["obs_std"], dtype=np.float32)
        self.trained_steps = data.get("trained_steps", 0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.mean) / np.maximum(self.std, 1e-6), -10.0, 10.0)
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
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--port", type=int, default=11040)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-ticks", type=int, default=20000)
    args = parser.parse_args()

    if args.controller == "policy":
        controller = PolicyController(Path(args.policy_path))
        label = f"policy ({controller.trained_steps:,} steps)"
    elif args.controller == "baseline":
        controller = baseline_controller
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
        args.godot, args.project, num_agents=args.agents, port=args.port, seed=args.seed
    ) as env:
        obs = env.reset()
        running_return = np.zeros(args.agents)
        running_length = np.zeros(args.agents, dtype=int)

        for _ in range(args.max_ticks):
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
                    returns.append(float(running_return[i]))
                    lengths.append(int(running_length[i]))
                    running_return[i] = 0.0
                    running_length[i] = 0
            if len(returns) >= args.episodes:
                break

    total = max(len(returns), 1)
    print(f"\n=== {label} ===")
    print(f"episodes            {len(returns)}")
    print(f"landing rate        {outcomes[LANDED] / total:.1%}")
    print(f"mean return         {np.mean(returns):.2f}")
    print(f"mean episode length {np.mean(lengths):.0f} ticks ({np.mean(lengths) / 60:.1f}s)")
    if impacts:
        print(f"touchdown speed     mean {np.mean(impacts):.2f}  median {np.median(impacts):.2f}"
              f"  p90 {np.percentile(impacts, 90):.2f} units/s")
    if landed_impacts:
        print(f"  of the landings   mean {np.mean(landed_impacts):.2f} units/s")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {OUTCOME_NAMES.get(outcome, outcome):<8} {count / total:6.1%}  ({count})")


if __name__ == "__main__":
    main()
