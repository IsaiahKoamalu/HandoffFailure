"""PPO for the Godot powered-descent task.

A single-file implementation, kept deliberately plain: an MLP actor-critic with
a state-independent diagonal Gaussian, GAE(lambda), a clipped surrogate, and
running observation normalisation. Nothing here is novel; the interesting part
of this project is the environment, and a standard algorithm makes it much
easier to tell an environment bug from an algorithm bug.

Two details that are easy to get wrong and matter a lot here:

* **Truncation is not termination.** An episode cut off at the step limit still
  has a future, so its return has to be bootstrapped from the value of the final
  observation. Treating the cut as a terminal state teaches the policy that time
  running out is like hitting the ground, which is exactly backwards.
* **Observation normalisation is part of the policy.** The running mean and
  variance are exported alongside the weights, because a policy fed raw
  observations in Godot would see inputs from a completely different
  distribution than it trained on.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evaluate import CONSTANT_STD, expert_for
from godot_env import ACT_SIZE, RUNNING, GodotVecEnv


class RunningNorm:
    """Welford running mean/variance over the observation batch."""

    def __init__(self, size: int) -> None:
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / self.scale(), -10.0, 10.0).astype(np.float32)

    # Per-channel divisor, with a floor that treats a channel which never moved
    # as unit scale rather than as something to divide by.
    #
    # CONSTANT_STD is not an epsilon guarding a divide by zero -- 1e-8 already
    # does that. It is a decision about what counts as constant. A channel whose
    # standard deviation over an entire training run is a thousandth of its own
    # range never varied, and dividing by that turns a rounding difference into
    # an input of a hundred and fifty. Measured: the shipped landing policy has
    # obs_std[19] = 1.0e-4 on the fuel channel, because it was trained with
    # unlimited tanks and the value was always exactly 1.0. Giving the player
    # ship a fuel budget then fed it 0.985, which normalised to -147 against
    # inputs that otherwise live in [-3, 3], and the policy stopped landing.
    def scale(self) -> np.ndarray:
        deviation = np.sqrt(self.var + 1e-8)
        return np.where(deviation < CONSTANT_STD, 1.0, deviation)


class ValueNorm:
    """Running mean/std of the value targets, so the critic learns in units of 1.

    The critic used to regress raw discounted returns, which run to about 130
    here with errors around 50 -- and value clipping bounded how far a
    prediction could move per update by ``--clip``, the *policy ratio* clip,
    0.2. In return units that is a leash 250 times shorter than the error, and
    ``clamp`` has no gradient once saturated, so any time the critic wanted to
    move more than 0.2 it received nothing at all. Value loss never converged in
    600k steps: 54, then 20, then back to 43.

    Normalising the targets puts them in units where 0.2 is a sensible step and
    the clip does the job it was written for. The network predicts a normalised
    value; everything outside this class works in real returns.

    Note the logged value_loss changes meaning with this: it is now roughly the
    fraction of return variance the critic fails to explain, so 1.0 is "no
    better than guessing the mean" and smaller is better.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, returns: torch.Tensor) -> None:
        batch = returns.detach().numpy().reshape(-1).astype(np.float64)
        batch_mean = float(batch.mean())
        batch_var = float(batch.var())
        batch_count = batch.size

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def normalise(self, real: torch.Tensor) -> torch.Tensor:
        return (real - self.mean) / self.std

    def denormalise(self, normalised: torch.Tensor) -> torch.Tensor:
        return normalised * self.std + self.mean


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        # The small final-layer gain matters: a policy that starts out saturating
        # its own action range spends the first several thousand steps flailing.
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_size), std=0.01),
        )
        # Start narrower than the usual -0.5. Thrust is direction-only once it
        # reaches the ship, so wide noise on a small mean command does not
        # explore around it, it replaces it.
        self.log_std = nn.Parameter(torch.full((act_size,), -1.0))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.critic(obs).squeeze(-1),
        )


def behaviour_clone(
    env: GodotVecEnv,
    model: ActorCritic,
    normaliser: RunningNorm,
    ticks: int,
    epochs: int,
    lr: float,
    rounds: int = 1,
    expert=None,
) -> np.ndarray:
    """Regress the actor onto the hand-written controller, with DAgger rounds.

    PPO from scratch spent five million steps and still landed less often than
    the proportional controller in evaluate.py. That is not really an algorithm
    failure: exploration has to discover the *structure* of a descent-rate hold
    and a great-circle cruise by random search, and a Gaussian around zero is a
    bad way to find either. So it is handed over instead.

    Plain cloning is not enough on its own, and the reason is worth stating
    because the symptom is misleading. Fitting the expert's actions on the
    expert's own states drives the training error down happily -- and the
    resulting policy still fails, because its small errors move it into states
    the expert never visited, where it has been taught nothing at all. Measured
    here: action MSE improved 2.5x with no change in landing rate, while the
    failures moved from ditching to hovering.

    DAgger (Ross, Gordon & Bagnell, 2011) fixes exactly that. After the first
    round the *learner* flies, and the expert labels the states the learner
    actually reaches. The dataset stops being a description of how the expert
    flies and becomes a description of how to recover from where the learner
    ends up, which is the thing that was missing.

    Returns the last observation batch, so the caller can carry straight on.
    """
    # Whose actions are being fitted. Defaulting to the landing law would put
    # the old bug back for any caller that forgets, so there is no default.
    if expert is None:
        expert_fn = expert_for(env.task)
    else:
        expert_fn = expert

    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    optimiser = torch.optim.Adam(model.actor.parameters(), lr=lr)

    obs = env.reset()
    for round_index in range(max(rounds, 1)):
        learner_drives = round_index > 0
        for tick in range(ticks):
            if tick % 60 == 0:
                env.set_hud(
                    phase=f"DAGGER {round_index + 1}/{rounds}" if learner_drives
                    else "CLONING BASELINE",
                    iter=tick,
                    steps=tick * env.num_agents,
                )
            # The expert labels every state regardless of who is flying.
            expert = expert_fn(obs)
            observations.append(obs.copy())
            targets.append(expert)
            normaliser.update(obs)

            if learner_drives:
                with torch.no_grad():
                    action = model.actor(
                        torch.as_tensor(normaliser(obs))
                    ).numpy()
                action = np.clip(action, -1.0, 1.0)
            else:
                action = expert
            obs, _, _, _, _ = env.step(action)

        x = torch.as_tensor(normaliser(np.concatenate(observations)))
        y = torch.as_tensor(np.concatenate(targets))
        count = x.shape[0]
        for epoch in range(epochs):
            order = torch.randperm(count)
            total = 0.0
            for start in range(0, count, 4096):
                batch = order[start : start + 4096]
                loss = ((model.actor(x[batch]) - y[batch]) ** 2).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)
            if epoch == epochs - 1:
                print(
                    f"  round {round_index + 1}/{rounds}  "
                    f"{count:,} samples  mse {total / count:.5f}",
                    flush=True,
                )

    return obs


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = GodotVecEnv(
        args.godot,
        args.project,
        num_agents=args.agents,
        port=args.port,
        seed=args.seed,
        render=args.render,
        render_rate=args.render_rate,
        repeat=args.repeat,
        fuel_seconds=args.fuel,
        # A rendered run is paced by the engine rather than by this side, so
        # a rollout takes far longer in wall-clock terms without anything
        # being wrong. The watchdog has to allow for that.
        step_timeout=120.0 if args.render else 30.0,
        task=args.task,
    )
    # The engine names the task's observation width in its hello message, so
    # the network is built to fit whatever task is actually running rather than
    # to a constant on this side that only happened to match one of them.
    obs_size = env.obs_size
    print(f"task '{env.task}': obs {obs_size}, act {env.act_size}", flush=True)

    model = ActorCritic(obs_size, ACT_SIZE, args.hidden).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    normaliser = RunningNorm(obs_size)
    value_norm = ValueNorm()

    n = args.agents
    rollout = args.rollout

    obs_buf = torch.zeros(rollout, n, obs_size)
    act_buf = torch.zeros(rollout, n, ACT_SIZE)
    logp_buf = torch.zeros(rollout, n)
    rew_buf = torch.zeros(rollout, n)
    val_buf = torch.zeros(rollout, n)
    # `cut` marks a frame an episode ended on, terminated or truncated alike.
    # One flag is enough: after either kind of ending the next frame belongs to
    # a new episode, so neither the bootstrap nor the advantage carry may cross
    # it. Truncation's genuine tail value is folded into that frame's reward
    # where it is computed, not carried here.
    cut_buf = torch.zeros(rollout, n)

    if args.warm_start > 0:
        print(
            f"cloning the '{env.task}' baseline controller "
            f"for {args.warm_start} ticks", flush=True)
        raw_obs = behaviour_clone(
            env, model, normaliser, args.warm_start, args.clone_epochs,
            args.clone_lr, args.dagger_rounds, expert_for(env.task),
        )
    else:
        raw_obs = env.reset()

    if args.warm_std is not None:
        # Cloning leaves a near-deterministic policy; PPO then samples around it
        # with whatever log_std happens to hold. At the default that is sigma
        # 0.37 against actions in [-1, 1] -- noise of the same order as the
        # commands themselves, which does not explore around a precision
        # landing so much as replace it. The first rollout after a warm start
        # is then bad data, and the first update learns from it.
        with torch.no_grad():
            model.log_std.fill_(math.log(args.warm_std))
        print(f"warm start: action sigma set to {args.warm_std:.3f}", flush=True)

    normaliser.update(raw_obs)
    next_obs = torch.as_tensor(normaliser(raw_obs))

    best_rate = -1.0
    best_step = 0
    episode_return = np.zeros(n)
    episode_length = np.zeros(n, dtype=np.int64)
    recent_returns: collections.deque[float] = collections.deque(maxlen=100)
    recent_lengths: collections.deque[int] = collections.deque(maxlen=100)
    recent_outcomes: collections.deque[int] = collections.deque(maxlen=200)
    recent_impacts: collections.deque[float] = collections.deque(maxlen=200)

    log_path = out_dir / "progress.csv"
    log_file = log_path.open("w", encoding="utf-8")
    # One column per outcome the *running* task declares, in the order it
    # declared them, rather than landing's six names spelled out. The header
    # therefore describes the file, which a fixed one stopped doing the moment
    # a second task existed -- docking's "collided" would have been written
    # under "escape_rate", or more likely not at all.
    outcome_columns = [name for _, name in sorted(env.outcome_names.items())]
    log_file.write(
        "step,iteration,return,length,"
        + ",".join(f"{name}_rate" for name in outcome_columns)
        + ",mean_impact,value_loss,policy_loss,entropy,explained_var,elapsed\n"
    )

    global_step = 0
    iterations = args.steps // (rollout * n)
    start_time = time.time()

    for iteration in range(1, iterations + 1):
        # Linear learning-rate decay. Cheap, and it reliably stops a policy that
        # has already found a landing from shaking itself apart later on.
        frac = 1.0 - (iteration - 1) / iterations
        optimiser.param_groups[0]["lr"] = frac * args.lr

        for t in range(rollout):
            obs_buf[t] = next_obs
            with torch.no_grad():
                action, logp, _, value = model.act(next_obs)
            act_buf[t] = action
            logp_buf[t] = logp
            val_buf[t] = value_norm.denormalise(value)

            raw_obs, reward, terminated, truncated, info = env.step(
                np.clip(action.numpy(), -1.0, 1.0)
            )
            global_step += n

            ended = terminated | truncated
            rew_buf[t] = torch.as_tensor(reward)
            cut_buf[t] = torch.as_tensor(ended.astype(np.float32))

            # Truncated episodes get their value bootstrapped from the frame
            # they were cut on, not from the first frame of the next episode.
            for index, final in info["final_obs"].items():
                if truncated[index]:
                    with torch.no_grad():
                        tail = value_norm.denormalise(
                            model.value(torch.as_tensor(normaliser(final[None]))[0])
                        )
                    rew_buf[t, index] += args.gamma * tail

            episode_return += reward
            episode_length += 1
            for i in range(n):
                outcome = int(info["outcome"][i])
                if outcome != RUNNING:
                    recent_outcomes.append(outcome)
                    # Touchdown speed relative to the surface. The single most
                    # legible number in the whole run: it is what separates a
                    # landing from a crater.
                    recent_impacts.append(float(info["impact"][i]))
                if ended[i]:
                    recent_returns.append(float(episode_return[i]))
                    recent_lengths.append(int(episode_length[i]))
                    episode_return[i] = 0.0
                    episode_length[i] = 0

            normaliser.update(raw_obs)
            next_obs = torch.as_tensor(normaliser(raw_obs))

        # --- GAE ------------------------------------------------------------
        with torch.no_grad():
            next_value = value_norm.denormalise(model.value(next_obs))
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(n)
            for t in reversed(range(rollout)):
                if t == rollout - 1:
                    following_value = next_value
                else:
                    following_value = val_buf[t + 1]
                # `cut` zeroes both the bootstrap and the advantage carry,
                # because after either kind of ending frame t+1 belongs to a
                # different episode -- the env auto-resets -- and its value says
                # nothing about this one's future.
                #
                # This used to mask the bootstrap with `done`, which is right
                # only if nothing else supplied the cut episode's tail. Something
                # did: the loop above already folds `gamma * V(final frame)` into
                # the reward on truncation. So a timeout was bootstrapped twice,
                # the second time from the first frame of the *next* episode --
                # and that frame is a fresh docking approach with a 120-point
                # bonus still ahead of it, so it scores high.
                #
                # So every timeout was paid a spurious bonus. Worth fixing, and
                # measured to be almost inert: a truncation is one frame per
                # episode, episodes here run 300-1500 decisions, and a rollout is
                # 256 decisions x 32 agents, so truncation frames are about 0.1%
                # of the batch -- and `cut` already stopped the bad value
                # propagating backwards past that single frame. Two docking runs
                # either side of this change tracked each other to the decimal
                # (27.3% vs 27.1% docked at iteration 20, 0.0% by 45). It is a
                # correctness fix, not the reason PPO loses to its own clone.
                #
                # `done` is still what decides whether a tail exists at all; it
                # just no longer decides this.
                delta = rew_buf[t] + args.gamma * following_value * (1.0 - cut_buf[t]) - val_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * (1.0 - cut_buf[t]) * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf
            # How much of the variation in the return the critic actually
            # accounts for. 1.0 is a perfect value function, 0.0 is no better
            # than predicting the mean, and negative is worse than that.
            #
            # Logged because it is the one number that separates the two
            # explanations for a fine-tune that decays: an optimiser drifting
            # off a good policy, or advantages that are mostly noise because
            # nothing here can predict a return. No episode in this task fits
            # inside a rollout -- they run 300 to 1500 decisions against 256 --
            # so every advantage rests on this estimate, and until now it was
            # not measured.
            residual = returns - val_buf
            explained = 1.0 - residual.var() / returns.var().clamp(min=1e-8)
            # Fold this batch in before normalising, so early iterations -- when
            # the running stats are still the identity -- move quickly.
            value_norm.update(returns)

        b_obs = obs_buf.reshape(-1, obs_size)
        b_act = act_buf.reshape(-1, ACT_SIZE)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        # The critic works in normalised units; advantages and returns stay real.
        b_ret = value_norm.normalise(returns.reshape(-1))
        b_val = value_norm.normalise(val_buf.reshape(-1))

        batch_size = rollout * n
        minibatch = batch_size // args.minibatches
        indices = np.arange(batch_size)

        value_loss = policy_loss = entropy_value = torch.tensor(0.0)
        for _ in range(args.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch):
                slice_ = indices[start : start + minibatch]
                _, new_logp, entropy, new_value = model.act(b_obs[slice_], b_act[slice_])
                ratio = (new_logp - b_logp[slice_]).exp()

                advantage = b_adv[slice_]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                policy_loss = torch.max(
                    -advantage * ratio,
                    -advantage * ratio.clamp(1 - args.clip, 1 + args.clip),
                ).mean()

                clipped_value = b_val[slice_] + (new_value - b_val[slice_]).clamp(
                    -args.clip, args.clip
                )
                value_loss = 0.5 * torch.max(
                    (new_value - b_ret[slice_]) ** 2, (clipped_value - b_ret[slice_]) ** 2
                ).mean()

                entropy_value = entropy.mean()
                if iteration <= args.critic_warmup:
                    # Value-only updates to start with.
                    #
                    # A cloned actor arrives already competent; the critic
                    # arrives random. Its advantages are noise, and the first
                    # PPO updates spend that noise dismantling the actor. This
                    # was not subtle when it happened: landings fell from 36%
                    # to 2.5% over 150 iterations while the training return
                    # fell with them. Letting the critic catch up first costs a
                    # few seconds and makes the fine-tune an improvement rather
                    # than a demolition.
                    loss = args.vf_coef * value_loss
                else:
                    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_value

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimiser.step()

        # --- Logging ---------------------------------------------------------
        counts = collections.Counter(recent_outcomes)
        total = max(len(recent_outcomes), 1)
        rates = {name: counts[code] / total for code, name in env.outcome_names.items()}
        success_rate = rates.get(env.success_name, 0.0)
        mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
        mean_length = float(np.mean(recent_lengths)) if recent_lengths else float("nan")
        mean_impact = float(np.mean(recent_impacts)) if recent_impacts else float("nan")
        elapsed = time.time() - start_time

        log_file.write(
            f"{global_step},{iteration},{mean_return:.3f},{mean_length:.1f},"
            + ",".join(f"{rates.get(name, 0.0):.4f}" for name in outcome_columns)
            + f",{mean_impact:.3f},"
            f"{value_loss.item():.4f},"
            f"{policy_loss.item():.4f},{entropy_value.item():.4f},"
            f"{explained.item():.4f},{elapsed:.1f}\n"
        )
        log_file.flush()

        # Cheap, and it rides along on the next action message, so there is no
        # reason to rate-limit it the way the console line is.
        env.set_hud(
            phase="PPO",
            iter=iteration,
            steps=global_step,
            ret=round(mean_return, 2),
            # The rate and its label travel together, so the overlay reads
            # "docked 61.2%" on a docking run without knowing what docking is.
            # It used to be sent as "land", which on any other task is a lie.
            success=round(success_rate, 4),
            goal=env.success_name,
        )

        if iteration % args.log_every == 0 or iteration == 1:
            # Success first and named, then every other outcome the task has,
            # so the line is readable without knowing which task is running.
            breakdown = "  ".join(
                f"{name} {rates.get(name, 0.0):5.1%}" for name in outcome_columns
            )
            print(
                f"iter {iteration:4d}/{iterations}  step {global_step:>9,}  "
                f"return {mean_return:8.2f}  len {mean_length:6.0f}  "
                f"{breakdown}  ev {explained.item():5.2f}  "
                f"{global_step / max(elapsed, 1e-6):,.0f} sps",
                flush=True,
            )

        # Keep the best policy as well as the last one. A fine-tuning run does
        # not always end at its peak -- this one climbed from 80% to 91.5% and
        # then gave several points back -- and saving only the final weights
        # throws away the very thing the run was asked to find out.
        if len(recent_outcomes) >= 100 and success_rate > best_rate:
            best_rate = success_rate
            best_step = global_step
            save(model, normaliser, out_dir, args, env, tag="best")

        if iteration % args.save_every == 0 or iteration == iterations:
            save(model, normaliser, out_dir, args, env)

    log_file.close()
    env.close()
    save(model, normaliser, out_dir, args, env)
    if best_rate >= 0.0:
        print(f"best {env.success_name} rate {best_rate:.1%} at step {best_step:,}, "
              f"kept alongside as policy_best.json")
    print(f"\nwrote {out_dir / 'policy.json'} and {log_path}")


def save(
    model: ActorCritic, normaliser: RunningNorm, out_dir: Path, args,
    env: GodotVecEnv, tag: str = "",
) -> None:
    suffix = f"_{tag}" if tag else ""
    torch.save(model.state_dict(), out_dir / f"policy{suffix}.pt")
    # Only the actor is exported. The critic exists to train the actor and has
    # no job at inference time.
    layers = []
    for module in model.actor:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "w": module.weight.detach().numpy().tolist(),  # [out][in]
                    "b": module.bias.detach().numpy().tolist(),
                }
            )
    payload = {
        "format": 1,
        "obs_size": env.obs_size,
        "act_size": env.act_size,
        # Which task these weights fly. A policy loaded against a different one
        # would still run and still look busy while flying something it was
        # never trained for, so the loader checks this.
        "task": env.task,
        "activation": "tanh",
        "layers": layers,
        "obs_mean": normaliser.mean.tolist(),
        "obs_std": np.sqrt(normaliser.var + 1e-8).tolist(),
        "trained_steps": args.steps,
        # Physics ticks each action was held for during training. The in-game
        # autopilot holds for the same number: a policy that learned to act at
        # 10 Hz and is asked to act at 60 will re-decide six times inside one
        # of its own intended commands.
        "repeat": args.repeat,
    }
    (out_dir / f"policy{suffix}.json").write_text(json.dumps(payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", default="rl/runs/landing")
    parser.add_argument("--task", default="landing")
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--port", type=int, default=11008)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--warm-start", type=int, default=0,
                        help="ticks of baseline demonstration to clone before PPO")
    parser.add_argument("--clone-epochs", type=int, default=8)
    parser.add_argument("--dagger-rounds", type=int, default=1,
                        help="rounds of expert relabelling; 1 is plain cloning")
    parser.add_argument("--clone-lr", type=float, default=1e-3)
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="iterations of value-only updates before the policy moves")
    parser.add_argument("--fuel", type=float, default=0.0,
                        help="seconds of full thrust per episode. 0 is unlimited.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="physics ticks to hold each action for. One decision "
                             "every N ticks. At 1 the discount horizon "
                             "(1/(1-gamma) = 333 decisions) covers 5.6 seconds of "
                             "an episode that runs 18 to 54, so the landing bonus "
                             "reaches the start of an episode discounted to a "
                             "quarter of a reward point and the agent optimises "
                             "shaping instead. 6 puts the episode inside the horizon.")
    parser.add_argument("--warm-std", type=float, default=None,
                        help="action standard deviation to start PPO from after a "
                             "warm start. The default init is exp(-1) = 0.37 on "
                             "actions bounded to [-1, 1], which is enough noise to "
                             "destroy a cloned landing policy on its first rollout "
                             "-- and the gradient is then computed on the wreckage. "
                             "Try 0.08 when fine-tuning a clone.")
    parser.add_argument("--render", action="store_true",
                        help="watch training in a window instead of running headless")
    parser.add_argument("--render-rate", type=int, default=180,
                        help="physics ticks per second while rendering; 60 is real time")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
