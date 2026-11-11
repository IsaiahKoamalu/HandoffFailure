"""Unit test for the observation normalizer.

    uv run python python/tests/test_normalizer.py
"""
import sys
import numpy as np
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
from normalizer import ObsNormalizer

from normalizer import CLIP, CONSTANT_STD, ObsNormalizer


def test_running_statistics_equal_the_statistics_of_everything_seen():
    rng = np.random.default_rng(0)
    batches = [rng.normal(loc=[3.0, -1.0, 0.0], scale=[2.0, 0.5, 10.0], size=(32, 3)) for _ in range(50)]
    norm = ObsNormalizer(3)
    for batch in batches:
        norm.update(batch)
    everything = np.concatenate(batches)
    assert norm.count == len(everything)
    assert np.allclose(norm.mean, everything.mean(axis=0), atol=1e-10), norm.mean
    assert np.allclose(norm.var, everything.var(axis=0), atol=1e-10), norm.var


def test_a_constant_channel_is_left_at_unit_scale():
    norm = ObsNormalizer(2)
    batch = np.column_stack([np.linspace(0.0, 1.0, 32), np.full(32, 1.0)])
    norm.update(batch)
    assert norm.std[1] <= CONSTANT_STD
    # Off by 0.015 from a channel that never moved: shifted, not divided by ~0.
    out = norm(np.array([[0.5, 0.985]]))
    assert np.isclose(out[0, 1], -0.015, atol=1e-6), out


def test_output_is_clamped():
    norm = ObsNormalizer(1)
    norm.update(np.linspace(-1.0, 1.0, 32).reshape(-1, 1))
    out = norm(np.array([[1e6], [-1e6]]))
    assert np.array_equal(out, [[CLIP], [-CLIP]]), out


def test_frozen_statistics_do_not_move():
    norm = ObsNormalizer(1)
    norm.update(np.linspace(0.0, 1.0, 32).reshape(-1, 1))
    before = (norm.mean.copy(), norm.var.copy(), norm.count)
    norm.frozen = True
    norm.update(np.full((32, 1), 100.0))
    assert np.array_equal(norm.mean, before[0]) and np.array_equal(norm.var, before[1])
    assert norm.count == before[2]


def test_output_is_float32_like_the_engine():
    norm = ObsNormalizer(2)
    norm.update(np.random.default_rng(1).normal(size=(32, 2)))
    assert norm(np.zeros((4, 2))).dtype == np.float32


if __name__ == "__main__":
    test_running_statistics_equal_the_statistics_of_everything_seen()
    test_a_constant_channel_is_left_at_unit_scale()
    test_output_is_clamped()
    test_frozen_statistics_do_not_move()
    test_output_is_float32_like_the_engine()
    print("PASS")
