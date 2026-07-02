from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from distreqx.bijectors import Planar

DIM = 3


def _numerical_log_det(fn, x):
    jac = jax.jacfwd(fn)(x)
    _, log_det = jnp.linalg.slogdet(jac)
    return log_det


class PlanarTest(TestCase):
    def setUp(self):
        self.bij = Planar(jr.key(0), dim=DIM, negative_slope=0.1)

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

    def test_invalid_negative_slope_raises(self):
        with self.assertRaises(ValueError):
            Planar(jr.key(0), dim=DIM, negative_slope=0.0)
        with self.assertRaises(ValueError):
            Planar(jr.key(0), dim=DIM, negative_slope=1.0)

    def test_different_keys_give_different_transforms(self):
        x = jr.normal(jr.key(1), (DIM,))
        p0 = Planar(jr.key(0), dim=DIM, negative_slope=0.1)
        p1 = Planar(jr.key(1), dim=DIM, negative_slope=0.1)
        self.assertFalse(jnp.allclose(p0.forward(x), p1.forward(x)))

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
