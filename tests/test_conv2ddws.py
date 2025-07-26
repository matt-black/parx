import jax
import jax.numpy as jnp
import jax.random as jr
import numpy
import torch

from parx._torch import PartialConv2d
from parx.conv import PartialConvDWS

jax.config.update("jax_platform_name", "cpu")


class PartialConvDWSTorch(torch.nn.Module):

    conv_pw: PartialConv2d
    conv_dw: PartialConv2d
    return_mask: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        dtype=None,
        return_mask: bool = False,
    ):
        super().__init__()
        self.conv_dw = PartialConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            bias,
            padding_mode,
            device=torch.device("cpu"),
            dtype=dtype,
            return_mask=True,
        )
        self.conv_pw = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=torch.device("cpu"),
            dtype=dtype,
            return_mask=return_mask,
        )

    def forward(self, x: torch.Tensor, mask_in: torch.Tensor):
        y, mask = self.conv_dw(x, mask_in)
        return self.conv_pw(y, mask)


def test_pconv2d_value1m() -> None:
    """Test that 2d conv w/o bias gives same value as PyTorch library."""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConvDWS(2, 1, 1, 3, 1, "same", use_bias=False, key=jr.key(1))
    clt = PartialConvDWSTorch(1, 1, 3, 1, padding=1, bias=False)
    clt.conv_dw.weight = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_dw.weight))
    )
    clt.conv_pw.weight = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_pw.weight))
    )
    out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    out_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    assert numpy.allclose(out_jax, out_pyt.detach().numpy())  # type: ignore


def test_pconv2d_value1m_bias() -> None:
    """Test that 2d conv w/o bias gives same value as PyTorch library."""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConvDWS(2, 1, 1, 3, 1, "same", use_bias=True, key=jr.key(1))
    clt = PartialConvDWSTorch(1, 1, 3, 1, padding=1, bias=True)
    clt.conv_dw.weight = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_dw.weight))
    )
    clt.conv_dw.bias = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_dw._bias)[0, 0, :])
    )
    clt.conv_pw.weight = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_pw.weight))
    )
    clt.conv_pw.bias = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_pw._bias)[0, 0, :])
    )
    out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    out_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    assert numpy.allclose(out_jax, out_pyt.detach().numpy())  # type: ignore


def test_pconv2d_mask1m() -> None:
    """Test that same mask is generated as PyTorch"""
    x = numpy.expand_dims(numpy.ones((5, 5), dtype=jnp.float32), 0)
    m = numpy.ones_like(x)
    m[0, 2, 2] = 0
    clj = PartialConvDWS(
        2, 1, 1, 3, 1, "same", use_bias=False, return_mask=True, key=jr.key(1)
    )
    clt = PartialConvDWSTorch(
        1, 1, 3, 1, padding=1, bias=False, return_mask=True
    )
    clt.conv_dw.weight = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_dw.weight))
    )
    clt.conv_pw.weight = torch.nn.Parameter(
        torch.tensor(numpy.asarray(clj.conv_pw.weight))
    )
    _, out_jax = clj(jnp.asarray(x), jnp.asarray(m))
    out_jax = numpy.asarray(out_jax)
    _, out_pyt = clt.forward(
        torch.tensor(x[None, ...]), mask_in=torch.tensor(m[None, ...])
    )
    assert numpy.allclose(out_jax, out_pyt.detach().numpy())  # type: ignore
