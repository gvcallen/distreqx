"""Distribution combining independent distributions over complementary
leaves of a shared pytree event."""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
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


def _pick(*leaves):
    """Return the first non-`None` leaf; `None` if they all are."""
    for leaf in leaves:
        if leaf is not None:
            return leaf
    return None


class Combined(
    AbstractProbDistribution,
    AbstractCDFDistribution,
    AbstractSurvivalDistribution,
):
    r"""Combines independent distributions over complementary partitions of
    one pytree event.

    Each distribution in `distributions` models a different, disjoint subset
    of a shared event pytree's leaves; every leaf it doesn't own is `None` in
    its own `event_shape`. Sampling and scoring merge/project across the
    whole tuple, so the combined distribution speaks in terms of the whole
    pytree even though each part only describes a fraction of it.

    Any two (or more) distributions whose `event_shape`s mirror one shared
    pytree structure -- `None` at every leaf they don't own -- can be
    combined, e.g. `a` owning `{"x": ..., "y": None}` and `b` owning
    `{"x": None, "y": ...}`:

    ```python
    dist = Combined((a, b))
    sample = dist.sample(key)  # {"x": ..., "y": ...}
    # log p({"x": x, "y": y}) == a.log_prob({"x": x, "y": None})
    #                          + b.log_prob({"x": None, "y": y})
    ```

    See `tests/combined_test.py` for a complete, runnable example.
    """

    distributions: tuple[AbstractDistribution, ...]

    def __init__(self, distributions: tuple[AbstractDistribution, ...]):
        """Initializes a Combined distribution.

        **Arguments:**

        - `distributions`: a tuple of two or more distributions, each over a
            disjoint partition of one shared event pytree. Every one's own
            event pytree (as reported by `event_shape`, `sample`, ...) must
            mirror all the others', holding `None` at every leaf it doesn't
            own and a real value at every leaf it does. A leaf may be `None`
            in all of them if it simply doesn't exist in this instance.
        """
        distributions = tuple(distributions)
        if len(distributions) < 2:
            raise ValueError("`Combined` requires at least two distributions.")
        self.distributions = distributions

        try:
            shapes = [d.event_shape for d in distributions]
            owner_counts = jtu.tree_map(
                lambda *s: sum(x is not None for x in s),
                *shapes,
                is_leaf=_is_none_or_shape,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Every distribution's `event_shape` must mirror the same "
                "pytree structure (e.g. matching dict keys or list lengths)."
            ) from e
        if not all(n <= 1 for n in jtu.tree_leaves(owner_counts)):
            raise ValueError(
                "At most one distribution may claim a value at any given "
                "leaf (a leaf that's `None` in all of them is fine and "
                "simply stays `None`)."
            )

    def _project(self, dist: AbstractDistribution, full: PyTree) -> PyTree:
        """Drop the leaves `dist` doesn't own from a full pytree."""
        return jtu.tree_map(
            lambda shape, v: v if shape is not None else None,
            dist.event_shape,
            full,
            is_leaf=_is_none_or_shape,
        )

    def _merge(self, parts: list[PyTree], is_leaf=_is_none) -> PyTree:
        """Combine complementary, `None`-padded pytrees into a full one.

        `is_leaf` must stop recursion at `None` and, when merging shape
        pytrees (as `event_shape` does), at shape tuples too -- otherwise
        `tree_map` would try to descend into a real shape tuple in one part
        to match a `None` in another.
        """
        return jtu.tree_map(_pick, *parts, is_leaf=is_leaf)

    def sample(self, key: Key[Array, ""]) -> PyTree:
        """Samples a full pytree event."""
        keys = jax.random.split(key, len(self.distributions))
        parts = [d.sample(k) for d, k in zip(self.distributions, keys)]
        return self._merge(parts)

    def sample_and_log_prob(self, key: Key[Array, ""]) -> tuple[PyTree, Array]:
        """Returns a full pytree sample and its log prob."""
        keys = jax.random.split(key, len(self.distributions))
        samples, log_probs = zip(
            *(d.sample_and_log_prob(k) for d, k in zip(self.distributions, keys))
        )
        return self._merge(list(samples)), jnp.sum(jnp.asarray(log_probs))

    def log_prob(self, value: PyTree) -> Array:
        """Sum of every part's log probability, each over its own leaves."""
        return jnp.sum(
            jnp.asarray(
                [d.log_prob(self._project(d, value)) for d in self.distributions]
            )
        )

    def entropy(self) -> Array:
        """Sum of every part's entropy."""
        return jnp.sum(jnp.asarray([d.entropy() for d in self.distributions]))

    def mean(self) -> PyTree:
        """Full pytree merging every part's mean."""
        return self._merge([d.mean() for d in self.distributions])

    def mode(self) -> PyTree:
        """Full pytree merging every part's mode."""
        return self._merge([d.mode() for d in self.distributions])

    def median(self) -> PyTree:
        """Full pytree merging every part's median."""
        return self._merge([d.median() for d in self.distributions])

    def variance(self) -> PyTree:
        """Full pytree merging every part's variance."""
        return self._merge([d.variance() for d in self.distributions])

    def stddev(self) -> PyTree:
        """Full pytree merging every part's standard deviation."""
        return self._merge([d.stddev() for d in self.distributions])

    def log_cdf(self, value: PyTree) -> Array:
        """Sum of every part's log CDF, each over its own leaves."""
        return jnp.sum(
            jnp.asarray(
                [d.log_cdf(self._project(d, value)) for d in self.distributions]
            )
        )

    def icdf(self, value: PyTree) -> PyTree:
        """Full pytree merging every part's ICDF."""
        return self._merge(
            [d.icdf(self._project(d, value)) for d in self.distributions]
        )

    @property
    def event_shape(self) -> PyTree:
        """A pytree of shapes mirroring the full event."""
        return self._merge(
            [d.event_shape for d in self.distributions], is_leaf=_is_none_or_shape
        )

    @property
    def support(self) -> tuple[PyTree, PyTree]:
        """See `Distribution.support`.

        Full pytrees merging every part's `(lower, upper)` bounds.
        """
        lowers, uppers = zip(*(d.support for d in self.distributions))
        return self._merge(list(lowers)), self._merge(list(uppers))

    def kl_divergence(self, other_dist, **kwargs) -> Array:
        r"""Calculates the KL divergence to another `Combined` distribution.

        Decomposes as the sum of each part's own KL divergence to the
        corresponding part of `other_dist`, mirroring `Joint.kl_divergence`.

        **Arguments:**

        - `other_dist`: Another `Combined` distribution, with the same
            number of parts and leaf partitioning.
        - `kwargs`: Forwarded to every part's `kl_divergence`.

        **Returns:**

        - `KL(self || other_dist)`.
        """
        if not isinstance(other_dist, Combined):
            raise TypeError(
                "KL divergence is only supported between two Combined " "distributions."
            )
        if len(self.distributions) != len(other_dist.distributions):
            raise ValueError(
                "The two Combined distributions do not have the same "
                "number of parts."
            )
        return jnp.sum(
            jnp.asarray(
                [
                    d.kl_divergence(od, **kwargs)
                    for d, od in zip(self.distributions, other_dist.distributions)
                ]
            )
        )
