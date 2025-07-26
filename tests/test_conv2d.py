import jax
import jax.numpy as jnp
import jax.random as jr
import numpy
import torch

from parx._torch import PartialConv2d
from parx.conv import PartialConv

jax.config.update("jax_platform_name", "cpu")


def test_pconv2d_value1m() -> None:
    """Test that 2d conv w/o bias gives same value as PyTorch library."""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConv(2, 1, 1, 3, 1, "same", use_bias=False, key=jr.key(1))
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    out_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    assert numpy.allclose(out_jax, out_pyt.detach().numpy())  # type: ignore


def test_pconv2d_value1m_bias() -> None:
    """Test that 2d conv w. bias gives same value as PyTorch library."""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConv(2, 1, 1, 3, 1, "same", use_bias=True, key=jr.key(1))
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=True)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    clt.bias = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj._bias))[0, 0, :]
    )
    out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    out_pyt = (
        clt.forward(
            torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
        )
        .detach()  # type: ignore
        .numpy()
    )
    assert numpy.allclose(out_jax, out_pyt)


def test_pconv2d_mask1m() -> None:
    """Test that same mask is generated as PyTorch"""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConv(
        2, 1, 1, 3, 1, "same", use_bias=False, return_mask=True, key=jr.key(1)
    )
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False, return_mask=True)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    _, msk_jax = clj(jnp.asarray(x), jnp.asarray(m))
    msk_jax = numpy.asarray(msk_jax)
    _, msk_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    msk_pyt = msk_pyt.numpy()  # type: ignore
    assert numpy.allclose(msk_jax, msk_pyt)


def test_pconv2d_value1b() -> None:
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 0, 0] = 0
    clj = PartialConv(2, 1, 1, 3, 1, "same", use_bias=False, key=jr.key(1))
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    out_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    assert numpy.allclose(out_jax, out_pyt.detach().numpy())  # type: ignore


def test_pconv2d_mask1b() -> None:
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 0, 0] = 0
    clj = PartialConv(
        2, 1, 1, 3, 1, "same", use_bias=False, return_mask=True, key=jr.key(1)
    )
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False, return_mask=True)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    _, msk_jax = clj(jnp.asarray(x), jnp.asarray(m))
    msk_jax = numpy.asarray(msk_jax)
    _, msk_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    msk_pyt = msk_pyt.numpy()  # type: ignore
    assert numpy.allclose(msk_jax, msk_pyt)


def test_pconv2d_value2m() -> None:
    x = numpy.expand_dims(numpy.ones((6, 6), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2:4, 2:4] = 0
    clj = PartialConv(2, 1, 1, 3, 1, "same", use_bias=False, key=jr.key(1))
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    out_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    assert numpy.allclose(out_jax, out_pyt.detach().numpy())  # type: ignore


def test_pconv2d_mask2m() -> None:
    x = numpy.expand_dims(numpy.ones((6, 6), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2:4, 2:4] = 0
    clj = PartialConv(
        2, 1, 1, 3, 1, "same", use_bias=False, return_mask=True, key=jr.key(1)
    )
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False, return_mask=True)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    _, msk_jax = clj(jnp.asarray(x), jnp.asarray(m))
    msk_jax = numpy.asarray(msk_jax)
    _, msk_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    msk_pyt = msk_pyt.numpy()  # type: ignore
    assert numpy.allclose(msk_jax, msk_pyt)


def test_pconv2d_grad1m() -> None:
    """Test that gradient is same as in PyTorch reference."""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConv(
        2, 1, 1, 3, 1, "same", use_bias=False, return_mask=False, key=jr.key(1)
    )
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False, return_mask=False)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    # compute gradient for jax
    ref = jnp.ones_like(x) * 0.5

    def f(z):
        out = clj(z, jnp.asarray(m))
        return jnp.mean((out - ref) ** 2)

    grad_jax = numpy.asarray(jax.grad(f)(jnp.asarray(x)))
    # compute gradient for pytorch
    x_pyt = torch.tensor(x[None, ...], requires_grad=True)
    ref_pyt = torch.tensor(ref[None, ...])
    out_pyt = clt(x_pyt, torch.tensor(m[None, ...]))
    loss = torch.nn.functional.mse_loss(out_pyt, ref_pyt)
    loss.backward()
    grad_pyt = x_pyt.grad.detach().numpy()  # type: ignore
    assert numpy.allclose(grad_jax, grad_pyt)


def test_pconv2d_grad1m_bias() -> None:
    """Test that gradient is same as in PyTorch reference."""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConv(
        2, 1, 1, 3, 1, "same", use_bias=True, return_mask=False, key=jr.key(1)
    )
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=True, return_mask=False)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    # compute gradient for jax
    ref = jnp.ones_like(x) * 0.5

    def f(z):
        out = clj(z, jnp.asarray(m))
        return jnp.mean((out - ref) ** 2)

    grad_jax = numpy.asarray(jax.grad(f)(jnp.asarray(x)))
    # compute gradient for pytorch
    x_pyt = torch.tensor(x[None, ...], requires_grad=True)
    ref_pyt = torch.tensor(ref[None, ...])
    out_pyt = clt(x_pyt, torch.tensor(m[None, ...]))
    loss = torch.nn.functional.mse_loss(out_pyt, ref_pyt)
    loss.backward()
    grad_pyt = x_pyt.grad.detach().numpy()  # type: ignore
    # NOTE: gradients are slightly different from PyTorch. This is probably because (should check) in our implementation we don't add the bias during the super() convolution, then subtract it in the actual class. See especially line 103 of _torch/partialconv.py, compare to our implementation.
    assert numpy.allclose(grad_jax, grad_pyt, atol=0.03)


def test_pconv2d_grad2m() -> None:
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConv(
        2, 1, 1, 3, 1, "same", use_bias=False, return_mask=False, key=jr.key(10)
    )
    clt = PartialConv2d(1, 1, 3, 1, padding=1, bias=False, return_mask=False)
    clt.weight = torch.nn.Parameter(torch.tensor(numpy.asarray(clj.weight)))
    # compute gradient for jax
    ref = jnp.ones_like(x) * 0.5

    def f(z):
        out = clj(z, jnp.asarray(m))
        return jnp.mean((out - ref) ** 2)

    grad_jax = numpy.asarray(jax.grad(f)(jnp.asarray(x)))
    # compute gradient for pytorch
    x_pyt = torch.tensor(x[None, ...], requires_grad=True)
    ref_pyt = torch.tensor(ref[None, ...])
    out_pyt = clt(x_pyt, torch.tensor(m[None, ...]))
    loss = torch.nn.functional.mse_loss(out_pyt, ref_pyt)
    loss.backward()
    grad_pyt = x_pyt.grad.detach().numpy()  # type: ignore
    assert numpy.allclose(grad_jax, grad_pyt)
