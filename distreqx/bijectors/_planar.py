import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from ._bijector import (
    AbstractBijector,
    AbstractForwardInverseBijector,
    AbstractFwdLogDetJacBijector,
    AbstractInvLogDetJacBijector,
)


class Planar(
    AbstractForwardInverseBijector,
    AbstractInvLogDetJacBijector,
    AbstractFwdLogDetJacBijector,
):
    r"""Planar bijector: $y = x + u \cdot \text{leaky\_relu}(w^T x + b)$.

    See [Rezende and Mohamed, 2015](https://arxiv.org/abs/1505.05770). The original
    paper uses a $\tanh$ activation, but that has no closed-form inverse; this uses a
    leaky ReLU instead (as the paper's appendix suggests), which does, at the cost of
    slightly less smooth transformed densities.
    """

    weight: Array
    act_scale: Array
    bias: Array
    negative_slope: float

    _is_constant_jacobian: bool = eqx.field(static=True, init=False)
    _is_constant_log_det: bool = eqx.field(static=True, init=False)

    def __init__(self, key: PRNGKeyArray, *, dim: int, negative_slope: float = 1e-2):
        """Initializes a `Planar` bijector.

        **Arguments:**

        - `key`: JAX random key used to initialise `w`, `u` and `b`.
        - `dim`: Dimension of the bijection.
        - `negative_slope`: The leaky ReLU's negative slope, in `(0, 1)`. Defaults to
            0.01.
        """
        if not 0 < negative_slope < 1:
            raise ValueError("`negative_slope` must be in (0, 1).")
        w_key, u_key, b_key = jr.split(key, 3)
        self.weight = 1e-2 * jr.normal(w_key, (dim,))
        self.act_scale = 1e-2 * jr.normal(u_key, (dim,))
        self.bias = 1e-2 * jr.normal(b_key, ())
        self.negative_slope = negative_slope
        self._is_constant_jacobian = False
        self._is_constant_log_det = False

    def _invertible_act_scale(self) -> Array:
        """Constrains `u` so that `w^T u >= -1`, guaranteeing invertibility.

        See Appendix A.1 of Rezende and Mohamed, 2015.
        """
        wtu = self.act_scale @ self.weight
        m_wtu = -1 + jnp.log1p(jax.nn.softplus(wtu))
        return self.act_scale + (m_wtu - wtu) * self.weight / jnp.sum(self.weight**2)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        u = self._invertible_act_scale()
        pre_act = x @ self.weight + self.bias
        act = jax.nn.leaky_relu(pre_act, negative_slope=self.negative_slope)
        y = x + u * act
        slope = jnp.where(pre_act < 0, self.negative_slope, 1.0)
        log_det = jnp.log(jnp.abs(1 + u @ (slope * self.weight)))
        return y, log_det

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|.

        Derivation: let $z = w^Tx + b$. Since $x = y - u \\cdot \\text{lrelu}(z)$,
        substituting into $z = w^Tx + b$ and solving gives
        $z = (w^Ty + b) / (1 + w^Tus)$, where $s$ is the leaky ReLU's slope at $z$
        (found from the sign of $w^Ty + b$, since the denominator is positive).
        """
        u = self._invertible_act_scale()
        numerator = self.weight @ y + self.bias
        slope = jnp.where(numerator < 0, self.negative_slope, 1.0)
        us = u * slope
        denominator = 1 + self.weight @ us
        x = y - us * (numerator / denominator)
        log_det = -jnp.log(jnp.abs(1 + us @ self.weight))
        return x, log_det

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return self is other
