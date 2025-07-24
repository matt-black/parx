"""Partial convolutions."""

import math
from collections.abc import Callable
from typing import Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.nn._misc import default_init
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Num, PRNGKeyArray

__all__ = ["PartialConv", "PartialConvBlock"]


class PartialConv(eqx.nn.Conv):
    """Partial convolution layer"""

    fixed: bool = eqx.field(static=True)
    return_mask: bool = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    # fft attrs
    is_fft: bool = eqx.field(static=True)
    # mask/masking attrs
    mask_update_kernel: Array
    update_mask_fun: Callable[[Array], Array]
    # bias term
    _bias: Array | None

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        return_mask: bool = False,
        fft_conv: bool = False,
        *,
        weight: Array | None = None,
        fixed: bool = False,
        key: PRNGKeyArray,
    ):
        """Initialize the layer.

        Args:
            num_spatial_dims (int): number of spatial dimensions
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int | Sequence[int]): size of the convolution kernel
            key (PRNGKeyArray): a `jax.random.PRNGKey` used to provide randomness for parameter initialization. (Keyword only argument).
            stride (int | Sequence[int], optional): stride of the convolution. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): padding of the convolution. Defaults to 0.
            dilation (int | Sequence[int], optional): dilation of the convolution. Defaults to 1.
            groups (int, optional): number of input channel groups. Defaults to 1.
            use_bias (bool, optional): whether to add on a bias after the convolution. Defaults to False.
            padding_mode (str, optional): string to specify padding values. See Equinox `nn.Conv` documentation. Defaults to "ZEROS".
            dtype (_type_, optional): dtype to use for the weight and bias in this layer. Defaults to None, which will use either `jnp.float32` or `jnp.float64` depending on whether JAX is in 64-bit mode.
            return_mask (bool, optional): return the current mask. Defaults to False.
            fft_conv (bool, optional): use FFT convolution. Defaults to False.
        """
        bkey, ckey = jax.random.split(key, 2)
        grouped_in_channels = in_channels // groups
        lim = 1 / math.sqrt(
            grouped_in_channels
            * (
                kernel_size
                if isinstance(kernel_size, int)
                else math.prod(kernel_size)
            )
        )
        super().__init__(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
            padding_mode,
            dtype,
            key=ckey,
        )
        # internal properties for my use
        self.fixed = fixed
        # NOTE: self.weight is set by Equinox, this is overwriting it.
        if weight is not None:
            self.weight = weight
        if fixed:
            self.weight = jax.lax.stop_gradient(self.weight)
        # fft specific properties
        self.return_mask = return_mask
        self.is_fft = fft_conv
        # masking properties for partial convolution
        upd_kernel_size = [out_channels, grouped_in_channels]
        if isinstance(self.kernel_size, int):
            upd_kernel_size += [
                self.kernel_size,
            ] * self.num_spatial_dims
        else:
            upd_kernel_size += list(self.kernel_size)
        self.mask_update_kernel = jax.lax.stop_gradient(
            jnp.ones_like(self.weight)
        )
        self.update_mask_fun = Partial(
            jax.lax.conv_general_dilated,
            rhs=self.mask_update_kernel,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=1,
        )
        self.window_size = math.prod(self.mask_update_kernel.shape[2:])
        # setup the bias term
        if use_bias:
            bshape = (out_channels,) + (1,) * num_spatial_dims
            self._bias = default_init(bkey, bshape, dtype, lim)
        else:
            self._bias = None

    def _fft_convolution(self, x: Array, x_fourier: bool) -> Array:
        fourier_axes = list(range(1, self.num_spatial_dims + 1))
        if not x_fourier:
            x_fft = jnp.fft.fftn(x, axes=fourier_axes)
        else:
            x_fft = x
        y_fft = jnp.multiply(x_fft, self.weight)
        if x_fourier:
            return y_fft
        else:
            return jnp.fft.ifftn(y_fft, axes=fourier_axes)

    def __call__(  # type: ignore
        self,
        x: Num[Array, "..."],
        mask: Bool[Array, "..."],
        epsilon: float = 1e-8,
    ) -> Union[Array, Tuple[Array, Array]]:
        """Forward pass of a partial convolution.

        Args:
            x (Num[Array]): input array.
            mask (Bool[Array]): mask array.
            epsilon (float, optional): small parameter to prevent division by zero. Defaults to 1e-8.

        Returns:
            Union[Array, Tuple[Array, Array]]: either the post-convolution array, or the post-convolution array and the updated mask, if `return_mask=True` was set during module initialization.
        """
        # compute updated mask & scaler (ratio)
        update_mask = self.update_mask_fun(mask[None, ...]).squeeze(0)
        mask_scaler = self.window_size / (update_mask + epsilon)
        update_mask = jnp.clip(update_mask, 0, 1)
        mask_scaler = jnp.multiply(mask_scaler, update_mask)
        # do the actual convolution, then add bias if applicable
        if self.is_fft:
            out = self._fft_convolution(jnp.multiply(x, mask), x_fourier=False)
        else:
            out = super().__call__(jnp.multiply(x, mask))

        if self._bias is not None:
            out = jnp.multiply(
                jnp.add(jnp.multiply(out, mask_scaler), self._bias), update_mask
            )
        else:
            out = jnp.multiply(out, mask_scaler)
        if self.return_mask:
            return out, update_mask
        else:
            return out


class PartialConvBlock(eqx.Module):
    """A two-layer block of partial convolutions.
    Often used in UNet-style architectures within the encoder.
    """

    conv1: PartialConv
    conv2: Optional[PartialConv]
    activation: Callable[[Array], Array]

    def __init__(
        self,
        num_spatial_dims: int,
        single_conv: bool,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: Union[str, int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        use_bias: bool = False,
        padding_mode: str = "ZEROS",
        dtype=None,
        fft_conv: bool = False,
        activation: str = "leaky_relu",
        *,
        key: PRNGKeyArray,
    ):
        """Initialize the block of convolutions.

        Args:
            num_spatial_dims (int): number of spatial dimensions
            single_conv (bool): only do a single convolution (instead of default, 2)
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of convolution kernel
            key (PRNGKeyArray): PRNG key
            stride (int | Sequence[int], optional): convolution stride, can be specified per-convolution. Defaults to 1.
            padding (Union[str, int, Sequence[int], Sequence[Tuple[int, int]]], optional): padding to apply before/after each spatial dimension. Defaults to 0.
            dilation (int | Sequence[int], optional): convolution dilation. Defaults to 1.
            groups (int, optional): groups for convolution. Defaults to 1.
            use_bias (bool, optional): whether or not to use a bias term. Defaults to False.
            padding_mode (str, optional): how to do the padding. Defaults to "ZEROS".
            dtype (_type_, optional): datatype for weights in the layer. Defaults to None.
            fft_conv (bool, optional): whether to use FFT convolutions or not. Defaults to False.
            activation (str, optional): the activation function to use after each convolution. Defaults to "leaky_relu".

        Raises:
            ValueError: if invalid activation function specified.
        """
        key1, key2 = jax.random.split(key, 2)
        self.conv1 = PartialConv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias,
            padding_mode,
            dtype,
            True,
            fft_conv,
            key=key1,
        )
        if not single_conv:
            self.conv2 = PartialConv(
                num_spatial_dims,
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                use_bias,
                padding_mode,
                dtype,
                True,
                fft_conv,
                key=key2,
            )
        else:
            self.conv2 = None
        if activation == "leaky_relu":
            self.activation = Partial(jax.nn.leaky_relu, negative_slope=0.1)
        elif activation == "relu":
            self.activation = jax.nn.relu
        else:
            raise ValueError("only ReLU and Leaky ReLU are valid")

    def __call__(self, x: Array, mask_in: Array) -> Tuple[Array, Array]:
        """Forward pass through the convolution block.

        Args:
            x (Array): input array
            mask_in (Array): mask array

        Returns:
            Tuple[Array,Array]: output array and updated mask
        """
        y, mask = self.conv1(x, mask_in)
        y = self.activation(y)
        if self.conv2 is not None:
            z, mask = self.conv2(x, mask)
            return self.activation(z), mask
        else:
            return y, mask
