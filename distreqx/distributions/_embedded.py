"""Distribution that embeds a base distribution's varying leaves into a fixed pytree."""

from typing import Optional

import jax.numpy as jnp
import jax.tree
from jaxtyping import Array, Key, PyTree

from ._distribution import (
    AbstractCDFDistribution,
    AbstractDistribution,
    AbstractProbDistribution,
    AbstractSurvivalDistribution,
)


def _is_none(x) -> bool:
    return x is None


def _is_none_or_shape(x) -> bool:
    # Shape tuples must also stop recursion, or `tree_map` would descend into
    # them and treat each dimension as its own leaf.
    return x is None or isinstance(x, tuple)


class Embedded(
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSurvivalDistribution,
):
    r"""A distribution over a subset of a PyTree's leaves, with the rest held fixed.

    It is common to want to describe the distribution over only some leaves
    in a PyTree, for example if some are considered random variables while
    others are considered fixed/static. `Embedded` lets a base distribution
    model only the random leaves, while the fixed leaves are added/removed
    appropriately during `sample`, `log_prob` and other relevant methods.

    Since the fixed leaves are constants rather than observations, each
    are considered to have zero variances, and contribute zero log probability.

    ```python
    import jax
    import jax.numpy as jnp
    from distreqx.distributions import Embedded, Joint, Normal

    varying = Joint({"a": Normal(0.0, 1.0), "b": None})
    dist = Embedded(varying, fixed={"a": None, "b": jnp.array(2.0)})

    sample = dist.sample(jax.random.key(0))  # {"a": ..., "b": Array(2.0)}
    dist.log_prob(sample)  # == varying.log_prob({"a": sample["a"], "b": None})
    ```

    Most methods forward to `distribution` and embed the result back into the
    full pytree (or, for `log_prob`, `log_cdf`, `icdf`, project the argument
    down to the varying leaves first); a base that doesn't implement one of
    them raises the same `NotImplementedError` it always would. `variance` and
    `stddev` are the exception, contributing `0` at the fixed leaves. `entropy`
    and `kl_divergence` range only over the varying leaves.

    !!! note

        `log_prob`, `log_cdf`, and `icdf` trust that the fixed leaves of their
        argument already match `fixed` and do not check them.
    """

    distribution: AbstractDistribution
    fixed: PyTree

    def __init__(self, distribution: AbstractDistribution, fixed: PyTree):
        """Initializes an Embedded distribution.

        **Arguments:**

        - `distribution`: the base distribution over the varying leaves. Its
            own event pytree (as reported by `event_shape`, `sample`, etc.)
            must mirror `fixed`, holding `None` at every fixed leaf and a real
            value at every varying leaf. A leaf may be `None` in both if it
            doesn't exist in this instance of the model.
        - `fixed`: the full event pytree holding the fixed values, with `None`
            at every varying leaf.
        """
        self.distribution = distribution
        self.fixed = fixed

        try:
            not_double_claimed = jax.tree.map(
                lambda d, f: (d is None) or (f is None),
                distribution.event_shape,
                fixed,
                is_leaf=_is_none_or_shape,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                "`distribution.event_shape` and `fixed` must mirror the same "
                "pytree structure (e.g. matching dict keys or list lengths)."
            ) from e
        if not all(jax.tree.leaves(not_double_claimed)):
            raise ValueError(
                "`distribution` and `fixed` must not both provide a value at "
                "the same leaf: at every leaf, at most one of "
                "`distribution.event_shape` and `fixed` may be non-`None` "
                "(fixed values come from `fixed`, varying ones from "
                "`distribution`; a leaf that's `None` in both is fine and "
                "simply stays `None`)."
            )

    def _embed(
        self, varying: PyTree, fill: Optional[PyTree] = None, is_leaf=_is_none
    ) -> PyTree:
        """Fill the varying values into a fixed structure -> full pytree."""
        if fill is None:
            fill = self.fixed
        return jax.tree.map(
            lambda v, f: v if v is not None else f, varying, fill, is_leaf=is_leaf
        )

    def _project(self, full: PyTree) -> PyTree:
        """Drop the fixed leaves from a full pytree -> varying pytree."""
        return jax.tree.map(
            lambda f, v: None if f is not None else v,
            self.fixed,
            full,
            is_leaf=_is_none,
        )

    def sample(self, key: Key[Array, ""]) -> PyTree:
        """Samples a full pytree event."""
        return self._embed(self.distribution.sample(key))

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[PyTree, Array]:
        """Returns a full pytree sample and its log prob."""
        varying, lp = self.distribution.sample_and_log_prob(key)
        return self._embed(varying), lp

    def log_prob(self, value: PyTree) -> Array:
        """Log probability of the varying leaves; fixed leaves are ignored."""
        return self.distribution.log_prob(self._project(value))

    def entropy(self) -> Array:
        """Entropy of the varying leaves; the fixed leaves add nothing."""
        return self.distribution.entropy()

    def mean(self) -> PyTree:
        """Full pytree with the base mean at the varying leaves."""
        return self._embed(self.distribution.mean())

    def mode(self) -> PyTree:
        """Full pytree with the base mode at the varying leaves."""
        return self._embed(self.distribution.mode())

    def median(self) -> PyTree:
        """Full pytree with the base median at the varying leaves."""
        return self._embed(self.distribution.median())

    def variance(self) -> PyTree:
        """Full pytree with the base variance at the varying leaves and `0` at
        the fixed leaves (constants don't vary)."""
        zeros = jax.tree.map(
            lambda f: None if f is None else jnp.zeros_like(f),
            self.fixed,
            is_leaf=_is_none,
        )
        return self._embed(self.distribution.variance(), fill=zeros)

    def stddev(self) -> PyTree:
        """Full pytree with the base standard deviation at the varying leaves
        and `0` at the fixed leaves."""
        return jax.tree.map(jnp.sqrt, self.variance())

    def log_cdf(self, value: PyTree) -> Array:
        """Log CDF of the varying leaves; fixed leaves are ignored."""
        return self.distribution.log_cdf(self._project(value))

    def icdf(self, value: PyTree) -> PyTree:
        """Full pytree with the base ICDF at the varying leaves and the fixed
        values at the fixed leaves."""
        return self._embed(self.distribution.icdf(self._project(value)))

    @property
    def event_shape(self) -> PyTree:
        """A pytree of shapes mirroring the full event.

        Relies on `distribution` reporting its own `event_shape` as a
        `None`-padded pytree mirroring `fixed`, which every pytree-shaped
        `distreqx` distribution does (see the class docstring).
        """
        fixed_shapes = jax.tree.map(
            lambda f: None if f is None else jnp.shape(f), self.fixed, is_leaf=_is_none
        )
        return self._embed(
            self.distribution.event_shape, fill=fixed_shapes, is_leaf=_is_none_or_shape
        )

    @property
    def support(self) -> tuple[PyTree, PyTree]:
        """See `Distribution.support`.

        Full pytrees with the base's `(lower, upper)` bounds at the varying
        leaves and the fixed values (their own, degenerate, single-point
        support) at the fixed leaves.
        """
        lower, upper = self.distribution.support
        return self._embed(lower), self._embed(upper)

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        r"""Calculates the KL divergence to another `Embedded` distribution.

        Decomposes as `KL(self.distribution || other_dist.distribution)` over
        the varying leaves, plus a fixed-leaf term that is `0` if `self.fixed`
        and `other_dist.fixed` agree everywhere and `inf` otherwise, mirroring
        `Deterministic.kl_divergence`, since a mismatched fixed leaf means the
        two distributions have disjoint support.

        **Arguments:**

        - `other_dist`: Another `Embedded` distribution, with the same
            varying/fixed leaf structure.
        - `kwargs`: Forwarded to `self.distribution.kl_divergence`.

        **Returns:**

        - `KL(self || other_dist)`.
        """
        if not isinstance(other_dist, Embedded):
            raise TypeError(
                "KL divergence is only supported between two Embedded " "distributions."
            )
        if jax.tree.structure(self.fixed, is_leaf=_is_none) != jax.tree.structure(
            other_dist.fixed, is_leaf=_is_none
        ):
            raise ValueError(
                "The two Embedded distributions do not share the same "
                "varying/fixed leaf structure."
            )

        agree = jax.tree.map(
            lambda a, b: True if a is None else jnp.array_equal(a, b),
            self.fixed,
            other_dist.fixed,
            is_leaf=_is_none,
        )
        all_agree = jnp.all(jnp.asarray(jax.tree.leaves(agree)))
        base_kl = self.distribution.kl_divergence(other_dist.distribution, **kwargs)
        return jnp.where(all_agree, base_kl, jnp.inf)
