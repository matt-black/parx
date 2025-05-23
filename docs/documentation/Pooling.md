# `pool`

Pooling layers that should be used in conjunction with partial convolution layers.
These take in a `mask` argument, as well, and applying pooling to the `mask` alongside the input array.

::: parx.pool.PartialPool
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: parx.pool.PartialMaxPool
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: parx.pool.PartialAvgPool
    handler: python
    options:
        show_source: false
        show_root_heading: true
