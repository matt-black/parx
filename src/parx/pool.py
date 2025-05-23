import math
from collections.abc import Callable
from typing import Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

__all__ = ["PartialPool", "PartialMaxPool", "PartialAvgPool"]


class PartialPool(eqx.nn.Pool):
    """Pooling layer to be used with partial convolutions.

    Takes in an input and a mask, and applies the pooling operation to both.
    """

    def __init__(
        self,
        init: Union[int, float, Array],
        operation: Callable[[Array, Array], Array],
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]],
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
    ):
        """Initialize the layer.

        Args:
            init (Union[int, float, Array]): initial value for the reduction
            operation (Callable[[Array, Array], Tuple[Array,Array]]): operation applied to the inputs of each window
            num_spatial_dims (int): # of spatial dimensions
            kernel_size (Union[int, Sequence[int]]): size of convolution kernel
            stride (Union[int, Sequence[int]]): stride of the convolution
            padding (Union[int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to apply before and after each spatial dimension. Defaults to 0.
            use_ceil (bool, optional): If True, use `ceil` to compute the final output shape instead of `floor`. Defaults to False.
        """
        super().__init__(
            init,
            operation,
            num_spatial_dims,
            kernel_size,
            stride,
            padding,
            use_ceil,
        )

    def __call__(  # type: ignore
        self, x: Array, mask: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Tuple[Array, Array]:
        """Forward pass of pooling operation.

        Args:
            x (Array): input array.
            mask (Array): mask array
            key (Optional[PRNGKeyArray], optional): Ignored; provided for compatibility with equinox API. Defaults to None.

        Returns:
            Tuple[Array, Array]: pooled input and mask.
        """
        x_out = super().__call__(x)
        mask_out = super().__call__(mask)
        return x_out, mask_out


class PartialMaxPool(PartialPool):
    """Max pooling layer to be used with partial convolutions."""

    def __init__(
        self,
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
    ):
        """Initialize a maximum pooling layer for masked inputs.

        Args:
            num_spatial_dims (int): # of spatial dimensions
            kernel_size (Union[int, Sequence[int]]): size of the convolution kernel
            stride (Union[int, Sequence[int]], optional): stride of the convolution. Defaults to 1.
            padding (Union[int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to add before and after each spatial dimension. Defaults to 0.
            use_ceil (bool, optional): If True, use `ceil` to compute the final output shape instead of `floor`. Defaults to False.
        """
        super().__init__(
            init=-jnp.inf,
            operation=jax.lax.max,
            num_spatial_dims=num_spatial_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_ceil=use_ceil,
        )


class PartialAvgPool(PartialPool):
    """Average pooling layer to be used with partial convolutions."""

    def __init__(
        self,
        num_spatial_dims: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[Tuple[int, int]]] = 0,
        use_ceil: bool = False,
    ):
        """Initialize a maximum pooling layer for masked inputs.

        Args:
            num_spatial_dims (int): # of spatial dimensions
            kernel_size (Union[int, Sequence[int]]): size of the convolution kernel
            stride (Union[int, Sequence[int]], optional): stride of the convolution. Defaults to 1.
            padding (Union[int, Sequence[int], Sequence[Tuple[int, int]]], optional): amount of padding to add before and after each spatial dimension. Defaults to 0.
            use_ceil (bool, optional): If True, use `ceil` to compute the final output shape instead of `floor`. Defaults to False.
        """
        super().__init__(
            init=0,
            operation=jax.lax.add,
            num_spatial_dims=num_spatial_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_ceil=use_ceil,
        )

    def __call__(
        self, x: Array, mask: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Tuple[Array, Array]:
        denom = math.prod(self.kernel_size)
        x_out, mask_out = super().__call__(x, mask)
        x_out = x_out / denom
        mask_out = (mask_out / denom) >= 0.5
        return x_out, mask_out
