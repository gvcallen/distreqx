"""Tests for `transformed.py`."""

from unittest import TestCase

import jax
import jax.numpy as jnp
import numpy as np
from parameterized import parameterized  # type: ignore

from distreqx.bijectors import ScalarAffine, Sigmoid
from distreqx.distributions import Normal, Transformed


class TransformedTest(TestCase):
    def setUp(self):
        self.seed = jax.random.key(1234)

    @parameterized.expand(
        [
            ("int16", jnp.array([0, 0], dtype=np.int16), Normal),
            ("int32", jnp.array([0, 0], dtype=np.int32), Normal),
            ("int64", jnp.array([0, 0], dtype=np.int64), Normal),
        ]
    )
    def test_integer_inputs(self, name, inputs, base_dist):
        base = base_dist(
            jnp.zeros_like(inputs, dtype=jnp.float32),
            jnp.ones_like(inputs, dtype=jnp.float32),
        )
        bijector = ScalarAffine(shift=jnp.array(0.0))
        dist = Transformed(base, bijector)

        log_prob = dist.log_prob(inputs)

        standard_normal_log_prob_of_zero = jnp.array(-0.9189385)
        expected_log_prob = jnp.full_like(
            inputs, standard_normal_log_prob_of_zero, dtype=jnp.float32
        )

        np.testing.assert_array_equal(log_prob, expected_log_prob)

    @parameterized.expand(
        [
            ("kl distreqx_to_distreqx", "distreqx_to_distreqx"),
        ]
    )
    def test_kl_divergence(self, name, mode_string):
        base_dist1 = Normal(
            loc=jnp.array([0.1, 0.5, 0.9]), scale=jnp.array([0.1, 1.1, 2.5])
        )
        base_dist2 = Normal(
            loc=jnp.array([-0.1, -0.5, 0.9]), scale=jnp.array([0.1, -1.1, 2.5])
        )
        bij_distreqx1 = ScalarAffine(shift=jnp.array(0.0))
        bij_distreqx2 = ScalarAffine(shift=jnp.array(0.0))
        distreqx_dist1 = Transformed(base_dist1, bij_distreqx1)
        distreqx_dist2 = Transformed(base_dist2, bij_distreqx2)

        expected_result_fwd = base_dist1.kl_divergence(base_dist2)
        expected_result_inv = base_dist2.kl_divergence(base_dist1)

        if mode_string == "distreqx_to_distreqx":
            result_fwd = distreqx_dist1.kl_divergence(distreqx_dist2)
            result_inv = distreqx_dist2.kl_divergence(distreqx_dist1)
        else:
            raise ValueError(f"Unsupported mode string: {mode_string}")

        np.testing.assert_allclose(result_fwd, expected_result_fwd, rtol=1e-2)
        np.testing.assert_allclose(result_inv, expected_result_inv, rtol=1e-2)

    def test_kl_divergence_on_same_instance_of_distreqx_bijector(self):
        base_dist1 = Normal(
            loc=jnp.array([0.1, 0.5, 0.9]), scale=jnp.array([0.1, 1.1, 2.5])
        )
        base_dist2 = Normal(
            loc=jnp.array([-0.1, -0.5, 0.9]), scale=jnp.array([0.1, -1.1, 2.5])
        )
        bij_distreqx = Sigmoid()
        distreqx_dist1 = Transformed(base_dist1, bij_distreqx)
        distreqx_dist2 = Transformed(base_dist2, bij_distreqx)
        expected_result_fwd = base_dist1.kl_divergence(base_dist2)
        expected_result_inv = base_dist2.kl_divergence(base_dist1)
        result_fwd = distreqx_dist1.kl_divergence(distreqx_dist2)
        result_inv = distreqx_dist2.kl_divergence(distreqx_dist1)
        np.testing.assert_allclose(result_fwd, expected_result_fwd, rtol=1e-2)
        np.testing.assert_allclose(result_inv, expected_result_inv, rtol=1e-2)

    def test_jittable(self):
        @jax.jit
        def f(x, d):
            return d.log_prob(x)

        base = Normal(jnp.array(0.0), jnp.array(1.0))
        bijector = ScalarAffine(jnp.array(0.0), jnp.array(1.0))
        dist = Transformed(base, bijector)
        x = jnp.zeros(())
        y = f(x, dist)
        self.assertIsInstance(y, jax.Array)

    @parameterized.expand(
        [
            ("increasing_positive_scale", 2.0, 1.0),
            ("decreasing_negative_scale", -2.0, 1.0),
        ]
    )
    def test_cdf_and_log_cdf(self, name, scale_val, shift_val):
        scale = jnp.array(scale_val)
        shift = jnp.array(shift_val)

        base_dist = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
        bijector = ScalarAffine(shift=shift, scale=scale)
        dist = Transformed(base_dist, bijector)

        # Test values in the transformed space Y
        y = jnp.array([-2.0, 0.0, 1.0, 3.5])

        # Inverse transform to base space X
        x = (y - shift) / scale

        # Calculate expected CDF based on monotonicity
        if scale_val > 0:
            expected_cdf = base_dist.cdf(x)
            expected_log_cdf = base_dist.log_cdf(x)
        else:
            expected_cdf = base_dist.survival_function(x)
            expected_log_cdf = base_dist.log_survival_function(x)

        np.testing.assert_allclose(dist.cdf(y), expected_cdf, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            dist.log_cdf(y), expected_log_cdf, rtol=1e-5, atol=1e-5
        )

    @parameterized.expand(
        [
            ("increasing_positive_scale", 2.0, -1.5),
            ("decreasing_negative_scale", -0.5, 3.0),
        ]
    )
    def test_icdf(self, name, scale_val, shift_val):
        scale = jnp.array(scale_val)
        shift = jnp.array(shift_val)

        base_dist = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
        bijector = ScalarAffine(shift=shift, scale=scale)
        dist = Transformed(base_dist, bijector)

        # Test probabilities
        p = jnp.array([0.01, 0.25, 0.5, 0.75, 0.99])

        # Calculate expected ICDF based on monotonicity
        if scale_val > 0:
            expected_icdf = scale * base_dist.icdf(p) + shift
        else:
            expected_icdf = scale * base_dist.icdf(1.0 - p) + shift

        np.testing.assert_allclose(dist.icdf(p), expected_icdf, rtol=1e-5, atol=1e-5)

    def test_median(self):
        base_dist = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))

        # Shifted by 5.0, scaled by -3.0
        # The base median is 0.0. The transformed median should be exactly the shift.
        bijector = ScalarAffine(shift=jnp.array(5.0), scale=jnp.array(-3.0))
        dist = Transformed(base_dist, bijector)

        expected_median = jnp.array(5.0)

        np.testing.assert_allclose(dist.median(), expected_median, rtol=1e-5, atol=1e-5)

    def test_cdf_methods_are_jittable(self):
        @jax.jit
        def f(val, prob, d):
            return d.cdf(val), d.log_cdf(val), d.icdf(prob), d.median()

        base = Normal(jnp.array(0.0), jnp.array(1.0))
        bijector = ScalarAffine(shift=jnp.array(1.0), scale=jnp.array(-1.0))
        dist = Transformed(base, bijector)

        val = jnp.array(0.5)
        prob = jnp.array(0.5)

        out_cdf, out_log_cdf, out_icdf, out_median = f(val, prob, dist)

        self.assertIsInstance(out_cdf, jax.Array)
        self.assertIsInstance(out_log_cdf, jax.Array)
        self.assertIsInstance(out_icdf, jax.Array)
        self.assertIsInstance(out_median, jax.Array)
