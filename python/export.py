"""Writing a trained policy as the policy.json that rl/neural_pilot.gd loads.

NeuralPilot runs the actor's mean in GDScript.
- Inputs are normalized with obs_mean, obs_std, tanh after every layer but the last.
- output gets clamped to [-1, 1]
- the critic and the exploration noise are only needed for training and are not exported.
"""

import json
import torch.nn as nn

FORMAT = 1

def export_policy(path, agent, normalizer, *, task, obs_names, repeat, trained_steps):
    modules = list(agent.actor_mean)
    linears = modules[0::2]
    expected = [nn.Linear, nn.Tanh] * (len(linears) - 1) + [nn.Linear]
    if [type(m) for m in modules] != expected:
        raise ValueError("NeuralPilot runs Linear, Tanh, ..., Linear; the actor has a different shape")

    state = normalizer.state()
    policy = {
        "format": FORMAT,
        "task": task,
        "obs_size": int(linears[0].in_features),
        "act_size": int(linears[-1].out_features),
        "activation": "tanh",
        "repeat": repeat,
        "trained_steps": trained_steps,
        "layers": [
            {"w": layer.weight.detach().cpu().tolist(), "b": layer.bias.detach().cpu().tolist()}
            for layer in linears
        ],
        "obs_mean": state["obs_mean"],
        "obs_std": state["obs_std"],
        "obs_names": obs_names,
        "normalizer_count": state["count"],
    }
    with open(path, "w") as f:
        json.dump(policy, f, indent=2)