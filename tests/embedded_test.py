"""Tests for `_embedded.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from distreqx.bijectors import Exp
from distreqx.distributions import Embedded, Joint, Normal, Transformed


class EmbeddedTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)
        # "a" is random (a scalar Normal); "b" is a frozen constant.
        self.varying = Joint({"a": Normal(jnp.array(0.0), jnp.array(1.0)), "b": None})
        self.fixed_value = jnp.array(2.0)
        self.dist = Embedded(self.varying, fixed={"a": None, "b": self.fixed_value})

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_invalid_parameters_mismatched_keys(self):
        # `fixed` has an extra key not present in `distribution`.
        with self.assertRaisesRegex(ValueError, "mirror the same pytree structure"):
            Embedded(
                Joint({"a": Normal(jnp.array(0.0), jnp.array(1.0))}),
                fixed={"a": None, "b": jnp.array(1.0)},
            )

    def test_invalid_parameters_both_sides_claim_a_leaf(self):
        # Both `distribution` and `fixed` claim "a" as their own.
        with self.assertRaisesRegex(ValueError, "must not both provide a value"):
            Embedded(
                Joint({"a": Normal(jnp.array(0.0), jnp.array(1.0)), "b": None}),
                fixed={"a": jnp.array(1.0), "b": jnp.array(1.0)},
            )

    def test_leaf_absent_from_both_sides_is_allowed(self):
        # "b" is `None` on both sides: it simply doesn't exist for this
        # instance of the model, and should stay `None` throughout.
        dist = Embedded(
            Joint({"a": Normal(jnp.array(0.0), jnp.array(1.0)), "b": None}),
            fixed={"a": None, "b": None},
        )
        sample = dist.sample(self.key)
        self.assertIsNone(sample["b"])
        self.assertIsNone(dist.event_shape["b"])
        self.assertIsNone(dist.mean()["b"])

    def test_sample_embeds_fixed_leaves(self):
        sample = self.dist.sample(self.key)
        self.assertEqual(set(sample.keys()), {"a", "b"})
        self.assertion_fn()(sample["b"], self.fixed_value)
        # "a" should match sampling directly from the wrapped base.
        expected_a = self.varying.sample(self.key)["a"]
        self.assertion_fn()(sample["a"], expected_a)

    def test_log_prob_ignores_fixed_leaves(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(1234.0)}  # "b" deliberately wrong
        expected = self.varying.log_prob({"a": jnp.array(0.3), "b": None})
        self.assertion_fn()(self.dist.log_prob(value), expected)

    def test_sample_and_log_prob_consistent_with_sample_then_log_prob(self):
        sample, lp = self.dist.sample_and_log_prob(self.key)
        self.assertion_fn()(lp, self.dist.log_prob(sample))
        self.assertion_fn()(sample["b"], self.fixed_value)

    def test_prob_is_exp_log_prob(self):
        value = {"a": jnp.array(0.3), "b": self.fixed_value}
        self.assertion_fn()(self.dist.prob(value), jnp.exp(self.dist.log_prob(value)))

    def test_entropy_matches_base(self):
        self.assertion_fn()(self.dist.entropy(), self.varying.entropy())

    def test_mean_mode_median_embed_fixed_value(self):
        for method in ("mean", "mode", "median"):
            result = getattr(self.dist, method)()
            self.assertion_fn()(result["a"], 0.0)
            self.assertion_fn()(result["b"], self.fixed_value)

    def test_variance_and_stddev_zero_at_fixed_leaves(self):
        variance = self.dist.variance()
        self.assertion_fn()(variance["a"], 1.0)
        self.assertion_fn()(variance["b"], 0.0)

        stddev = self.dist.stddev()
        self.assertion_fn()(stddev["a"], 1.0)
        self.assertion_fn()(stddev["b"], 0.0)

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, {"a": (), "b": ()})

    def test_event_shape_with_non_scalar_fixed_leaf(self):
        varying = Joint({"a": Normal(jnp.zeros(2), jnp.ones(2)), "b": None})
        dist = Embedded(varying, fixed={"a": None, "b": jnp.zeros(3)})
        self.assertEqual(dist.event_shape, {"a": (2,), "b": (3,)})

    def test_support_embeds_fixed_value(self):
        lower, upper = self.dist.support
        self.assertion_fn()(lower["a"], -jnp.inf)
        self.assertion_fn()(upper["a"], jnp.inf)
        self.assertion_fn()(lower["b"], self.fixed_value)
        self.assertion_fn()(upper["b"], self.fixed_value)

    def test_cdf_log_cdf_survival_function_project_fixed_leaves(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(1234.0)}
        expected_log_cdf = self.varying.log_cdf({"a": jnp.array(0.3), "b": None})
        self.assertion_fn()(self.dist.log_cdf(value), expected_log_cdf)
        self.assertion_fn()(self.dist.cdf(value), jnp.exp(expected_log_cdf))
        self.assertion_fn()(
            self.dist.survival_function(value), 1.0 - jnp.exp(expected_log_cdf)
        )

    def test_icdf_embeds_fixed_value(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(1234.0)}
        result = self.dist.icdf(value)
        expected_a = self.varying.icdf({"a": jnp.array(0.3), "b": None})["a"]
        self.assertion_fn()(result["a"], expected_a)
        self.assertion_fn()(result["b"], self.fixed_value)

    def test_kl_divergence_matches_base_when_fixed_leaves_agree(self):
        other_varying = Joint({"a": Normal(jnp.array(1.0), jnp.array(1.0)), "b": None})
        other = Embedded(other_varying, fixed={"a": None, "b": self.fixed_value})
        expected = self.varying.kl_divergence(other_varying)
        self.assertion_fn()(self.dist.kl_divergence(other), expected)

    def test_kl_divergence_is_infinite_when_fixed_leaves_disagree(self):
        other_varying = Joint({"a": Normal(jnp.array(1.0), jnp.array(1.0)), "b": None})
        other = Embedded(other_varying, fixed={"a": None, "b": jnp.array(3.0)})
        self.assertEqual(self.dist.kl_divergence(other), jnp.inf)

    def test_kl_divergence_wrong_type_raises(self):
        with self.assertRaisesRegex(TypeError, "only supported between two Embedded"):
            self.dist.kl_divergence(self.varying)

    def test_kl_divergence_mismatched_structure_raises(self):
        other_varying = Joint({"a": Normal(jnp.array(0.0), jnp.array(1.0)), "c": None})
        other = Embedded(other_varying, fixed={"a": None, "c": self.fixed_value})
        with self.assertRaisesRegex(ValueError, "same varying/fixed leaf structure"):
            self.dist.kl_divergence(other)

    def test_fixed_leaves_are_ordinary_differentiable_pytree_leaves(self):
        def f(fixed_val):
            dist = Embedded(self.varying, fixed={"a": None, "b": fixed_val})
            return dist.mean()["b"]

        grad = jax.grad(f)(jnp.array(2.0))
        self.assertEqual(grad, 1.0)

    def test_fully_varying_wraps_base_transparently(self):
        # `fixed=None` means every leaf is varying: `Embedded` should behave as
        # a transparent passthrough around the base distribution.
        base = Transformed(Normal(jnp.array(0.0), jnp.array(1.0)), Exp())
        dist = Embedded(base, fixed=None)

        self.assertEqual(dist.event_shape, base.event_shape)
        sample = dist.sample(self.key)
        self.assertion_fn()(sample, base.sample(self.key))
        self.assertion_fn()(dist.log_prob(sample), base.log_prob(sample))

    def test_base_not_implemented_errors_propagate(self):
        # `Exp` has a non-constant Jacobian, so `Transformed` cannot provide
        # `mean`/`mode`/`entropy`; several other stats are unconditionally
        # unimplemented on `Transformed`. `Embedded` should not paper over any
        # of this -- it should raise exactly what the base raises.
        base = Transformed(Normal(jnp.array(0.0), jnp.array(1.0)), Exp())
        dist = Embedded(base, fixed=None)

        with self.assertRaises(NotImplementedError):
            dist.mean()
        with self.assertRaises(NotImplementedError):
            dist.mode()
        with self.assertRaises(NotImplementedError):
            dist.entropy()
        with self.assertRaises(NotImplementedError):
            dist.median()
        with self.assertRaises(NotImplementedError):
            dist.variance()
        with self.assertRaises(NotImplementedError):
            dist.stddev()
        with self.assertRaises(NotImplementedError):
            dist.icdf(jnp.array(0.5))
        with self.assertRaises(NotImplementedError):
            dist.log_cdf(jnp.array(0.5))
        with self.assertRaises(NotImplementedError):
            dist.cdf(jnp.array(0.5))
        with self.assertRaises(NotImplementedError):
            _ = dist.support

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist, key):
            return dist.sample_and_log_prob(key)

        sample, log_prob = f(self.dist, self.key)
        self.assertIsInstance(sample["a"], jax.Array)
        self.assertIsInstance(sample["b"], jax.Array)
        self.assertIsInstance(log_prob, jax.Array)

        expected_sample, expected_log_prob = self.dist.sample_and_log_prob(self.key)
        self.assertion_fn()(sample["a"], expected_sample["a"])
        self.assertion_fn()(log_prob, expected_log_prob)
