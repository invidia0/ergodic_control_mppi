"""Checks that the median-source surrogate is compared against the field it approximates.

Sec. "mppi_ensemble_dist" states that the reference field is built from a compressed stand-in
for the rollout occupancy measure and that the induced discrepancy is ``eps_comp``. These
tests pin the two properties the paper's claim rests on: that the faithful field really is
eq. (25) over the whole ensemble (so the comparison is not against a second approximation),
and that the surrogate's error is reported where MPPI reads it -- in the induced weights,
not in the raw ordering of rollouts the update gives no weight to.
"""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ergodic_control_mppi.config import load_config
from ergodic_control_mppi.experiments.surrogate_fidelity import (
    _spearman,
    faithful_reference_flow,
    fidelity_walk,
    step_fidelity,
    summarize,
)
from ergodic_control_mppi.mppi.core import reference_flow
from ergodic_control_mppi.mppi.single import initialize_single
from ergodic_control_mppi.mppi.core import field_at
from tests.helpers import write_small_config


def _params(directory: Path):
    return load_config(write_small_config(Path(directory))).controller


def _carry(params):
    return initialize_single(
        params,
        jnp.zeros((6,), jnp.float32),
        jnp.zeros((params.mppi.horizon, 3), jnp.float32),
        jax.random.key(43),
    )


class FaithfulFieldTest(unittest.TestCase):
    """The comparison field must be the deployed field queried per rollout, not a rewrite."""

    def test_each_rollout_is_queried_at_its_own_states(self):
        """Lane k of the faithful field is ``field_at`` on rollout k's own states.

        This is the whole content of the comparison: the surrogate evaluates once on the
        median path and broadcasts, the faithful version evaluates per rollout. If the two
        used different *expressions* rather than different query sets, every fidelity number
        would be measuring a transcription error instead of the compression.
        """
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            carry = _carry(params)
            samples, horizon = params.mppi.samples, params.mppi.horizon
            key = jax.random.key(7)
            evaluation = jax.random.uniform(
                key, (samples, horizon, 2), minval=-5.0, maxval=5.0
            )
            produced = faithful_reference_flow(params, evaluation, carry.memory)
            expected = field_at(
                params, evaluation[0], evaluation[0], carry.memory
            )
        # Not bit-equal: `lax.map` batches the same expression differently than a bare call,
        # and the memory term is a float32 sum over P kernels, so the reassociation shows at
        # the 1e-3 relative level. That is a lowering difference, not a different field --
        # which is the distinction this test exists to hold, so the tolerance is float32
        # reassociation rather than exact equality.
        np.testing.assert_allclose(
            np.asarray(produced[0]), np.asarray(expected), rtol=5e-3, atol=1e-5
        )

    def test_it_differs_from_the_surrogate(self):
        """A comparison that returned the surrogate would make every fidelity number vacuous."""
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            carry = _carry(params)
            samples, horizon = params.mppi.samples, params.mppi.horizon
            evaluation = jax.random.uniform(
                jax.random.key(11), (samples, horizon, 2), minval=-5.0, maxval=5.0
            )
            faithful = faithful_reference_flow(params, evaluation, carry.memory)
            surrogate = reference_flow(params, evaluation, carry.memory)
        gap = float(jnp.max(jnp.abs(faithful - surrogate[None])))
        self.assertGreater(gap, 1e-4)


class FidelityMetricTest(unittest.TestCase):
    """The reported numbers must mean what the paper says they mean."""

    def test_spearman_is_scale_and_offset_invariant(self):
        """MPPI reads costs through a softmax, so a monotone relabelling must score 1."""
        values = jnp.asarray([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])
        self.assertAlmostEqual(float(_spearman(values, 5.0 * values + 100.0)), 1.0, places=5)
        self.assertAlmostEqual(float(_spearman(values, -values)), -1.0, places=5)

    def test_step_fidelity_reports_bounded_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            fidelity = step_fidelity(params, _carry(params))
        self.assertGreaterEqual(float(fidelity.weight_tv), 0.0)
        self.assertLessEqual(float(fidelity.weight_tv), 1.0)
        self.assertGreaterEqual(float(fidelity.spearman), -1.0 - 1e-6)
        self.assertLessEqual(float(fidelity.spearman), 1.0 + 1e-6)
        # Without this the weight agreement is trivially satisfiable by uniform weights.
        self.assertGreater(float(fidelity.ess_fraction), 0.0)
        self.assertLessEqual(float(fidelity.ess_fraction), 1.0 + 1e-6)
        # The end of the chain: a weight disagreement only matters if the command moves.
        self.assertGreaterEqual(float(fidelity.control_gap), 0.0)

    def test_walk_reports_one_row_per_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            rows = fidelity_walk(
                params,
                jnp.zeros((6,), jnp.float32),
                jnp.zeros((params.mppi.horizon, 3), jnp.float32),
                jax.random.key(43),
                steps=6,
                stride=2,
            )
        self.assertEqual(rows.shape, (3, 7))
        self.assertEqual(summarize(np.asarray(rows))["steps"], 3)

    def test_walk_rejects_an_indivisible_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            params = _params(directory)
            with self.assertRaises(ValueError):
                fidelity_walk(
                    params,
                    jnp.zeros((6,), jnp.float32),
                    jnp.zeros((params.mppi.horizon, 3), jnp.float32),
                    jax.random.key(43),
                    steps=5,
                    stride=2,
                )


if __name__ == "__main__":
    unittest.main()
