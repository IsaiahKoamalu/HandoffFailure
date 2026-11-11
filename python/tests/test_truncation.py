"""Unit  test for the truncation fix (time limit should not be an ending)

One ship with four decisions, gamma 0.9, lambda 0.5. The episode is truncated at step 1
with V(final_obs) = 4.0, and step 2 starts a new episode.
"""

import sys
import numpy as np
import torch

from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
from ppo import bootstrap_truncated, compute_gae

GAMMA = 0.9
LAMBDA = 0.5

# Shaped (num_steps, num_envs) = (4, 1), as in the training loop.
VALUES = torch.tensor([[2.0], [3.0], [0.5], [1.0]])
RAW_REWARDS = np.array([1.0, 1.0, 0.0, 2.0], dtype=np.float32)
TRUNCATED_AT = 1
FINAL_VALUE = 4.0
# dones[t] means "obs[t] is the first frame of a new episode".
DONES = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
NEXT_VALUE = torch.tensor([[1.5]])
NEXT_DONE = torch.tensor([0.0])

CORRECT = torch.tensor([[2.42], [1.6], [1.4575], [2.35]])
CLEANRL_AS_SHIPPED = torch.tensor([[0.8], [-2.0], [1.4575], [2.35]])

def advantages_for(rewards):
    """Helper that runs GAE"""
    rewards = torch.tensor(rewards).reshape(4, 1)
    advantages, returns = compute_gae(
        rewards, 
        VALUES, 
        DONES, 
        NEXT_VALUE, 
        NEXT_DONE, 
        GAMMA, 
        LAMBDA
    )
    assert torch.allclose(returns, advantages + VALUES), "returns must be advantages + values"
    return advantages

def test_bootstrapped_advantages_match_the_hand_worked_numbers():
    rewards = RAW_REWARDS.copy()
    # The rollout calls bootstrap_truncated once per step, across all ships;
    # here there is one ship, so step 1 is a batch of one.
    rewards[TRUNCATED_AT:TRUNCATED_AT + 1] = bootstrap_truncated(
        rewards[TRUNCATED_AT:TRUNCATED_AT + 1],
        np.array([True]),
        np.array([FINAL_VALUE], dtype=np.float32),
        GAMMA,
    )
    assert np.isclose(rewards[TRUNCATED_AT], 4.6), rewards
    advantages = advantages_for(rewards)
    assert torch.allclose(advantages, CORRECT, atol=1e-6), advantages.flatten()

def test_without_the_bootstrap_the_answer_is_cleanrls_and_wrong():
    advantages = advantages_for(RAW_REWARDS)
    assert torch.allclose(advantages, CLEANRL_AS_SHIPPED, atol=1e-6), advantages.flatten()
    assert not torch.allclose(advantages, CORRECT, atol=1e-6), "the test cannot tell the bug apart"

def test_bootstrap_touches_only_truncated_ships():
    reward = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    truncations = np.array([False, False, True])  # ship 1 terminated, ship 2 truncated
    final_values = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    out = bootstrap_truncated(reward, truncations, final_values, GAMMA)
    assert np.allclose(out, [1.0, 1.0, 7.3]), out
    assert np.array_equal(reward, [1.0, 1.0, 1.0]), "the caller's reward was modified"

if __name__ == "__main__":
    test_bootstrapped_advantages_match_the_hand_worked_numbers()
    test_without_the_bootstrap_the_answer_is_cleanrls_and_wrong()
    test_bootstrap_touches_only_truncated_ships()
    print("PASS")
