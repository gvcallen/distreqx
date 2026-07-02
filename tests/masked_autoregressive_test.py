from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from distreqx.bijectors import MaskedAutoregressive

DIM = 4


def _numerical_log_det(fn, x):
    jac = jax.jacfwd(fn)(x)
    _, log_det = jnp.linalg.slogdet(jac)
    return log_det


class MaskedAutoregressiveTest(TestCase):
    def setUp(self):
        self.bij = MaskedAutoregressive(jr.key(0), dim=DIM, nn_width=16, nn_depth=1)

    def assertion_fn(self, rtol=1e-5, atol=1e-5):
        return lambda x, y: np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

    def test_round_trip(self):
        x = jr.normal(jr.key(1), (DIM,))
        y, fwd_log_det = self.bij.forward_and_log_det(x)
        x_rec, inv_log_det = self.bij.inverse_and_log_det(y)
        self.assertion_fn(atol=1e-4)(x_rec, x)
        self.assertion_fn(atol=1e-4)(fwd_log_det, -inv_log_det)

    def test_log_det_matches_autodiff_jacobian(self):
        x = jr.normal(jr.key(1), (DIM,))
        _, log_det = self.bij.forward_and_log_det(x)
        numerical = _numerical_log_det(self.bij.forward, x)
        self.assertion_fn(atol=1e-4)(log_det, numerical)

    def test_jacobian_is_lower_triangular(self):
        # The defining property of an autoregressive transform: y[i] depends only
        # on x[:i+1], so dy[i]/dx[j] == 0 for j > i.
        x = jr.normal(jr.key(1), (DIM,))
        jac = jax.jacfwd(self.bij.forward)(x)
        self.assertion_fn(atol=1e-6)(jnp.triu(jac, k=1), jnp.zeros_like(jac))

    def test_dim_one_is_unconditional(self):
        # With nothing earlier to condition on, dimension 0's transform must be a
        # fixed (input-independent) affine map.
        bij = MaskedAutoregressive(jr.key(0), dim=1, nn_width=8, nn_depth=1)
        shift0, scale0 = bij._shift_and_scale(bij._conditioner(jnp.array([0.0])))
        shift1, scale1 = bij._shift_and_scale(bij._conditioner(jnp.array([5.0])))
        self.assertion_fn()(shift0, shift1)
        self.assertion_fn()(scale0, scale1)

    def test_flags(self):
        self.assertFalse(self.bij.is_constant_jacobian)
        self.assertFalse(self.bij.is_constant_log_det)

    def test_jittable(self):
        @eqx.filter_jit
        def f(bij, x):
            return bij.forward_and_log_det(x)

        x = jr.normal(jr.key(1), (DIM,))
        y, logdet = f(self.bij, x)
        self.assertIsInstance(y, jax.Array)
        self.assertIsInstance(logdet, jax.Array)
