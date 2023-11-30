description: Packs <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> components into a DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.pack" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.pack

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Packs <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> components into a DTensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.pack(
    tensors: Sequence[Any],
    layout: <a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>
) -> Any
</code></pre>



<!-- Placeholder for "Used in" -->

Packing and unpacking are inverse operations:

```
* unpack(pack(tensors)) == tensors
* pack(unpack(dtensor)) == dtensor
```

1. For any DTensor on the mesh, `unpack` returns the raw components placed on
   each underlying device.
2. Packing these raw components in the same order using `pack` returns a
   DTensor which should be identical to the original DTensor--both the content
   value and the layout.

**Shape, Rank, and Scalars**: The rank of the DTensor is the same as the
rank of its raw components, i.e., rank is preserved.  This leads to a
consistent interpretation for packing scalar values into a DTensor. The only
valid layout for a scalar value is fully replicated, and the individual
components must be identical scalars.

Each input `tensors[i]` will be copied to `layout.mesh.local_device[i]`
if not already on the local device. Non-local components should not be passed
to `pack`; use `copy_to_mesh` and `relayout` to place tensors on all global
devices on a mesh.

It is the caller's responsibility to ensure that the underlying values
for `pack` adhere to the specified layout, and that only as many values are
specified as there are local devices. Pack does not move data between clients.
See examples below for more detail about layouts.

For example, assume we have a mesh `[X(2), Y(3)]`, which has in total 6
underlying devices. Futuremore, assume that the device location mapping is
the following:

```
device_ID  |  location X, Y
        0     0, 0
        1     0, 1
        2     0, 2
        3     1, 0
        4     1, 1
        5     1, 2
```

1. For 1-D vector DTensor with shape `[128]` with layout `[mesh.X]` and value
   as `range(128)`, the raw components will have shape `[64]` each, and the
   raw components will be:

   ```
   device_ID  |  raw component
           0     range(0, 64)
           1     range(0, 64)
           2     range(0, 64)
           3     range(64, 128)
           4     range(64, 128)
           5     range(64, 128)
   ```

   This also means for a 1-D DTensor with shape `[2]` and layout `[mesh.X]`,
   the raw components have shape `[1]` rather than the shape for scalar values
   `[]`.

2. For 2-D vector DTensor with shape `[2, 3]` with layout `[mesh.X, mesh.Y]`
   and value as `range(6)`, this is basically a fully-sharded DTensor.

   From global view, the content looks like
   ```
   [
     [0.0, 1.0, 2.0],
     [3.0, 4.0, 5.0],
   ]
   ```

   The raw components will have shape `[1, 1]` each, and have the following
   content:

   ```
   device_ID  |  raw component
           0     [[0.0]]
           1     [[1.0]]
           2     [[2.0]]
           3     [[3.0]]
           4     [[4.0]]
           5     [[5.0]]
   ```

3. For a scalar value `123.0` DTensor, it can only have one legitimate layout
   `[]` (no dimension, but fully replicated).

   The raw components will have shape `[]` each, and have the following
   content:

   ```
   device_ID  |  raw component
           0     123.0
           1     123.0
           2     123.0
           3     123.0
           4     123.0
           5     123.0
   ```

   Again, caller of `pack` is expected to provide 6 identical value raw
   components with scalar shapes.

4. For 3-D vector DTensor with shape `[2, 2, 3]` with layout
   `[X, unsharded, unsharded]` and value as `range(12)`,

   From global view, the content looks like:
   ```
   [
     [
       [0.0, 1.0, 2.0],
       [3.0, 4.0, 5.0],
     ],
     [
       [6.0, 7.0, 8.0],
       [9.0, 10., 11.],
     ],
   ]
   ```

   The raw components will have shape `[1, 2, 3]` each, and have the following
   content:

   ```
   device_ID  |  raw component
           0     range(6).reshape([1, 2, 3])
           1     range(6).reshape([1, 2, 3])
           2     range(6).reshape([1, 2, 3])
           3     range(6, 12).reshape([1, 2, 3])
           4     range(6, 12).reshape([1, 2, 3])
           5     range(6, 12).reshape([1, 2, 3])
   ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensors`<a id="tensors"></a>
</td>
<td>
The list of local tensor components to pack into a DTensor.
</td>
</tr><tr>
<td>
`layout`<a id="layout"></a>
</td>
<td>
The layout of the DTensor to be created.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A DTensor created from the individual component tensors.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
When `pack` is not called eagerly.
</td>
</tr>
</table>

