"""Scale bijector."""

from jax import numpy as jnp
from jaxtyping import Array

from ._bijector import AbstractBijector
from ._scalar_affine import AbstractScalarAffine


class Scale(AbstractScalarAffine):
    r"""Bijector that scales its input elementwise.

    The bijector is defined as follows:

    - Forward: $y = \text{scale} \times x$
    - Forward Jacobian determinant: $\log|\det J(x)| = \sum \log|\text{scale}|$
    - Inverse: $x = y / \text{scale}$
    - Inverse Jacobian determinant: $\log|\det J(y)| = -\sum \log|\text{scale}|$

    where `scale` parameterizes the bijector.
    """

    shift: Array
    scale: Array
    inv_scale: Array
    log_scale: Array
    _is_constant_jacobian: bool
    _is_constant_log_det: bool

    def __init__(self, scale: Array):
        """Initializes a `Scale` bijector.

        **Arguments:**

        - `scale`: the bijector's scale parameter. Must be non-zero.
        """
        self._is_constant_jacobian = True
        self._is_constant_log_det = True
        self.shift = jnp.zeros_like(scale)
        self.scale = scale
        self.inv_scale = 1.0 / scale
        self.log_scale = jnp.log(jnp.abs(scale))

    def same_as(self, other: AbstractBijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        if type(other) is Scale:
            return self.scale is other.scale
        return False
