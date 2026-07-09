"""Tests for `_combined.py`."""

from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from distreqx.bijectors import Sigmoid
from distreqx.distributions import (
    AbstractCDFDistribution,
    AbstractDistribution,
    AbstractProbDistribution,
    AbstractSurvivalDistribution,
    Combined,
    Normal,
    Transformed,
    Uniform,
)


def _is_dist(x) -> bool:
    return isinstance(x, AbstractDistribution)


class _DictJoint(
    AbstractProbDistribution, AbstractCDFDistribution, AbstractSurvivalDistribution
):
    """A minimal joint-over-a-dict-of-distributions, used only to build test
    fixtures for `Combined`. Deliberately not `distreqx.distributions.Joint`
    (which depends on more than bare upstream `distreqx` provides) -- this
    test file is kept dependency-free relative to bare upstream `distreqx`,
    matching `_combined.py` itself.
    """

    distributions: dict

    def __init__(self, distributions: dict):
        leaves = jtu.tree_leaves(distributions, is_leaf=_is_dist)
        if not leaves:
            raise ValueError("`distributions` must contain at least one distribution.")
        self.distributions = distributions

    @property
    def event_shape(self):
        return jtu.tree_map(
            lambda d: d.event_shape, self.distributions, is_leaf=_is_dist
        )

    @property
    def support(self):
        lower = jtu.tree_map(
            lambda d: d.support[0], self.distributions, is_leaf=_is_dist
        )
        upper = jtu.tree_map(
            lambda d: d.support[1], self.distributions, is_leaf=_is_dist
        )
        return lower, upper

    def sample(self, key):
        leaves, treedef = jtu.tree_flatten(self.distributions, is_leaf=_is_dist)
        keys = jtu.tree_unflatten(treedef, jax.random.split(key, len(leaves)))
        return jtu.tree_map(
            lambda d, k: d.sample(k), self.distributions, keys, is_leaf=_is_dist
        )

    def sample_and_log_prob(self, key):
        sample = self.sample(key)
        return sample, self.log_prob(sample)

    def log_prob(self, value):
        log_probs = jtu.tree_map(
            lambda d, v: d.log_prob(v), self.distributions, value, is_leaf=_is_dist
        )
        return jnp.sum(jnp.asarray(jtu.tree_leaves(log_probs)))

    def entropy(self):
        entropies = jtu.tree_map(
            lambda d: d.entropy(), self.distributions, is_leaf=_is_dist
        )
        return jnp.sum(jnp.asarray(jtu.tree_leaves(entropies)))

    def mean(self):
        return jtu.tree_map(lambda d: d.mean(), self.distributions, is_leaf=_is_dist)

    def median(self):
        return jtu.tree_map(lambda d: d.median(), self.distributions, is_leaf=_is_dist)

    def mode(self):
        return jtu.tree_map(lambda d: d.mode(), self.distributions, is_leaf=_is_dist)

    def variance(self):
        return jtu.tree_map(
            lambda d: d.variance(), self.distributions, is_leaf=_is_dist
        )

    def stddev(self):
        return jtu.tree_map(lambda d: d.stddev(), self.distributions, is_leaf=_is_dist)

    def log_cdf(self, value):
        log_cdfs = jtu.tree_map(
            lambda d, v: d.log_cdf(v), self.distributions, value, is_leaf=_is_dist
        )
        return jnp.sum(jnp.asarray(jtu.tree_leaves(log_cdfs)))

    def icdf(self, value):
        return jtu.tree_map(
            lambda d, v: d.icdf(v), self.distributions, value, is_leaf=_is_dist
        )

    def kl_divergence(self, other_dist, **kwargs):
        if not isinstance(other_dist, _DictJoint):
            raise TypeError(
                "KL divergence is only supported between two _DictJoint distributions."
            )
        kls = jtu.tree_map(
            lambda d1, d2: d1.kl_divergence(d2, **kwargs),
            self.distributions,
            other_dist.distributions,
            is_leaf=_is_dist,
        )
        return jnp.sum(jnp.asarray(jtu.tree_leaves(kls)))


class CombinedTest(TestCase):
    def setUp(self):
        self.key = jax.random.key(0)
        # Three independent parts, each owning one leaf of a shared
        # {"a", "b", "c"} pytree -- genuinely exercises the N-way (not just
        # two-way) generalization.
        self.a = _DictJoint(
            {"a": Normal(jnp.array(0.0), jnp.array(1.0)), "b": None, "c": None}
        )
        self.b = _DictJoint(
            {"a": None, "b": Uniform(jnp.array(3.8), jnp.array(4.2)), "c": None}
        )
        self.c = _DictJoint(
            {"a": None, "b": None, "c": Normal(jnp.array(5.0), jnp.array(2.0))}
        )
        self.dist = Combined((self.a, self.b, self.c))

    def assertion_fn(self, rtol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol)

    def test_requires_at_least_two_distributions(self):
        with self.assertRaisesRegex(ValueError, "at least two distributions"):
            Combined((self.a,))

    def test_invalid_parameters_mismatched_keys(self):
        # `b` has an extra key not present in `a`.
        with self.assertRaisesRegex(ValueError, "mirror the same pytree structure"):
            Combined(
                (
                    _DictJoint({"a": Normal(jnp.array(0.0), jnp.array(1.0))}),
                    _DictJoint(
                        {"a": None, "b": Uniform(jnp.array(0.0), jnp.array(1.0))}
                    ),
                )
            )

    def test_invalid_parameters_two_parts_claim_a_leaf(self):
        with self.assertRaisesRegex(ValueError, "At most one distribution"):
            Combined(
                (
                    _DictJoint(
                        {"a": Normal(jnp.array(0.0), jnp.array(1.0)), "b": None}
                    ),
                    _DictJoint(
                        {"a": Normal(jnp.array(0.0), jnp.array(1.0)), "b": None}
                    ),
                )
            )

    def test_leaf_absent_from_all_parts_is_allowed(self):
        # "c" is `None` in both `self.a` and `self.b`: it simply doesn't
        # exist for this (two-part) combination, and should stay `None`
        # throughout.
        dist = Combined((self.a, self.b))
        sample = dist.sample(self.key)
        self.assertIsNone(sample["c"])
        self.assertIsNone(dist.event_shape["c"])
        self.assertIsNone(dist.mean()["c"])

    def test_sample_merges_all_parts(self):
        sample = self.dist.sample(self.key)
        self.assertEqual(set(sample.keys()), {"a", "b", "c"})
        keys = jax.random.split(self.key, 3)
        self.assertion_fn()(sample["a"], self.a.sample(keys[0])["a"])
        self.assertion_fn()(sample["b"], self.b.sample(keys[1])["b"])
        self.assertion_fn()(sample["c"], self.c.sample(keys[2])["c"])

    def test_log_prob_sums_all_parts(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(4.0), "c": jnp.array(4.5)}
        expected = (
            self.a.log_prob({"a": jnp.array(0.3), "b": None, "c": None})
            + self.b.log_prob({"a": None, "b": jnp.array(4.0), "c": None})
            + self.c.log_prob({"a": None, "b": None, "c": jnp.array(4.5)})
        )
        self.assertion_fn()(self.dist.log_prob(value), expected)

    def test_sample_and_log_prob_consistent_with_sample_then_log_prob(self):
        sample, lp = self.dist.sample_and_log_prob(self.key)
        self.assertion_fn()(lp, self.dist.log_prob(sample))

    def test_prob_is_exp_log_prob(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(4.0), "c": jnp.array(4.5)}
        self.assertion_fn()(self.dist.prob(value), jnp.exp(self.dist.log_prob(value)))

    def test_entropy_sums_all_parts(self):
        expected = self.a.entropy() + self.b.entropy() + self.c.entropy()
        self.assertion_fn()(self.dist.entropy(), expected)

    def test_mean_merges_all_parts(self):
        # `Uniform` (used by `self.b`) doesn't implement `mode`/`median`, so
        # only `mean` is exercised across all three parts here.
        result = self.dist.mean()
        self.assertion_fn()(result["a"], self.a.mean()["a"])
        self.assertion_fn()(result["b"], self.b.mean()["b"])
        self.assertion_fn()(result["c"], self.c.mean()["c"])

    def test_median_merges_all_parts(self):
        result = self.dist.median()
        self.assertion_fn()(result["a"], self.a.median()["a"])
        self.assertion_fn()(result["b"], self.b.median()["b"])
        self.assertion_fn()(result["c"], self.c.median()["c"])

    def test_mode_propagates_base_limitations(self):
        # `Uniform.mode` (used by `self.b`) raises `NotImplementedError`; a
        # `Combined` distribution including it should raise the same, not
        # silently paper over it.
        with self.assertRaises(NotImplementedError):
            self.dist.mode()

    def test_variance_and_stddev_merge_all_parts(self):
        variance = self.dist.variance()
        self.assertion_fn()(variance["a"], self.a.variance()["a"])
        self.assertion_fn()(variance["b"], self.b.variance()["b"])
        self.assertion_fn()(variance["c"], self.c.variance()["c"])

        stddev = self.dist.stddev()
        self.assertion_fn()(stddev["a"], self.a.stddev()["a"])
        self.assertion_fn()(stddev["b"], self.b.stddev()["b"])
        self.assertion_fn()(stddev["c"], self.c.stddev()["c"])

    def test_event_shape(self):
        self.assertEqual(self.dist.event_shape, {"a": (), "b": (), "c": ()})

    def test_support_merges_all_parts(self):
        lower, upper = self.dist.support
        expected_a_lower, expected_a_upper = self.a.support
        expected_b_lower, expected_b_upper = self.b.support
        self.assertion_fn()(lower["a"], expected_a_lower["a"])
        self.assertion_fn()(upper["a"], expected_a_upper["a"])
        self.assertion_fn()(lower["b"], expected_b_lower["b"])
        self.assertion_fn()(upper["b"], expected_b_upper["b"])

    def test_cdf_log_cdf_survival_function_project_each_part(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(4.0), "c": jnp.array(4.5)}
        expected_log_cdf = (
            self.a.log_cdf({"a": jnp.array(0.3), "b": None, "c": None})
            + self.b.log_cdf({"a": None, "b": jnp.array(4.0), "c": None})
            + self.c.log_cdf({"a": None, "b": None, "c": jnp.array(4.5)})
        )
        self.assertion_fn()(self.dist.log_cdf(value), expected_log_cdf)
        self.assertion_fn()(self.dist.cdf(value), jnp.exp(expected_log_cdf))
        self.assertion_fn()(
            self.dist.survival_function(value), 1.0 - jnp.exp(expected_log_cdf)
        )

    def test_icdf_merges_all_parts(self):
        value = {"a": jnp.array(0.3), "b": jnp.array(0.4), "c": jnp.array(0.5)}
        result = self.dist.icdf(value)
        expected_a = self.a.icdf({"a": jnp.array(0.3), "b": None, "c": None})["a"]
        expected_b = self.b.icdf({"a": None, "b": jnp.array(0.4), "c": None})["b"]
        self.assertion_fn()(result["a"], expected_a)
        self.assertion_fn()(result["b"], expected_b)

    def test_kl_divergence_matches_sum_of_parts(self):
        other_a = _DictJoint(
            {"a": Normal(jnp.array(1.0), jnp.array(1.0)), "b": None, "c": None}
        )
        other = Combined((other_a, self.b, self.c))
        expected = self.a.kl_divergence(other_a)
        self.assertion_fn()(self.dist.kl_divergence(other), expected)

    def test_kl_divergence_wrong_type_raises(self):
        with self.assertRaisesRegex(TypeError, "only supported between two Combined"):
            self.dist.kl_divergence(self.a)

    def test_kl_divergence_mismatched_number_of_parts_raises(self):
        other = Combined((self.a, self.b))
        with self.assertRaisesRegex(ValueError, "same number of parts"):
            self.dist.kl_divergence(other)

    def test_two_way_combination_still_works(self):
        # The generalized tuple design must still behave exactly like the
        # original two-distribution case.
        dist = Combined((self.a, self.b))
        value = {"a": jnp.array(0.3), "b": jnp.array(4.0), "c": None}
        expected = self.a.log_prob(
            {"a": jnp.array(0.3), "b": None, "c": None}
        ) + self.b.log_prob({"a": None, "b": jnp.array(4.0), "c": None})
        self.assertion_fn()(dist.log_prob(value), expected)

    def test_two_parts_fully_covering_a_pytree(self):
        # Two parts that between them fully own every leaf of a shared
        # pytree, with no leftover `None`-only leaves.
        base_x = Transformed(Normal(jnp.array(0.0), jnp.array(1.0)), Sigmoid())
        base_y = Transformed(Normal(jnp.array(1.0), jnp.array(1.0)), Sigmoid())
        a = _DictJoint({"x": base_x, "y": None})
        b = _DictJoint({"x": None, "y": base_y})
        dist = Combined((a, b))

        self.assertEqual(
            dist.event_shape, {"x": base_x.event_shape, "y": base_y.event_shape}
        )
        sample = dist.sample(self.key)
        keys = jax.random.split(self.key, 2)
        self.assertion_fn()(sample["x"], a.sample(keys[0])["x"])
        self.assertion_fn()(sample["y"], b.sample(keys[1])["y"])

    def test_base_not_implemented_errors_propagate(self):
        # `Sigmoid` has a non-constant Jacobian, so `Transformed` cannot
        # provide `mean`/`mode`/`entropy`; several other stats are
        # unconditionally unimplemented on `Transformed`. `Combined` should
        # not paper over any of this -- it should raise exactly what the
        # bases raise.
        base_x = Transformed(Normal(jnp.array(0.0), jnp.array(1.0)), Sigmoid())
        base_y = Transformed(Normal(jnp.array(1.0), jnp.array(1.0)), Sigmoid())
        a = _DictJoint({"x": base_x, "y": None})
        b = _DictJoint({"x": None, "y": base_y})
        dist = Combined((a, b))

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
            dist.icdf({"x": jnp.array(0.5), "y": jnp.array(0.5)})
        with self.assertRaises(NotImplementedError):
            dist.log_cdf({"x": jnp.array(0.5), "y": jnp.array(0.5)})
        with self.assertRaises(NotImplementedError):
            dist.cdf({"x": jnp.array(0.5), "y": jnp.array(0.5)})
        with self.assertRaises(NotImplementedError):
            _ = dist.support

    def test_jittable(self):
        @eqx.filter_jit
        def f(dist, key):
            return dist.sample_and_log_prob(key)

        sample, log_prob = f(self.dist, self.key)
        self.assertIsInstance(sample["a"], jax.Array)
        self.assertIsInstance(sample["b"], jax.Array)
        self.assertIsInstance(sample["c"], jax.Array)
        self.assertIsInstance(log_prob, jax.Array)

        expected_sample, expected_log_prob = self.dist.sample_and_log_prob(self.key)
        self.assertion_fn()(sample["a"], expected_sample["a"])
        self.assertion_fn()(log_prob, expected_log_prob)
