description: Returns a distributed TPU mesh optimized for AllReduce ring reductions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.create_tpu_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.create_tpu_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/tpu_util.py">View source</a>



Returns a distributed TPU mesh optimized for AllReduce ring reductions.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.create_tpu_mesh(
    mesh_dim_names: List[str],
    mesh_shape: List[int],
    mesh_name: str,
    ring_dims: Optional[int] = None,
    ring_axes: Optional[List[str]] = None,
    ring_bounds: Optional[List[int]] = None,
    can_split_host_across_rings: bool = True,
    build_ring_across_rings: bool = False,
    rotate_ring_across_rings: bool = False,
    use_xla_spmd: bool = layout_lib.USE_XLA_SPMD
) -> <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Only as many as leading axes specified by `ring_axes` as necessary will be
used to build rings, as long as the subslice formed by these axes have enough
cores to contain a ring of the required size. The leftover axes in `ring_axes`
won't affect results.

This function always uses all TPU devices, and offers more customization than
<a href="../../../tf/experimental/dtensor/create_distributed_mesh.md"><code>tf.experimental.dtensor.create_distributed_mesh</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh_dim_names`<a id="mesh_dim_names"></a>
</td>
<td>
List of mesh dimension names.
</td>
</tr><tr>
<td>
`mesh_shape`<a id="mesh_shape"></a>
</td>
<td>
Shape of the mesh.
</td>
</tr><tr>
<td>
`mesh_name`<a id="mesh_name"></a>
</td>
<td>
A unique name for the mesh. If empty, internally generate one.
</td>
</tr><tr>
<td>
`ring_dims`<a id="ring_dims"></a>
</td>
<td>
Optional; The number of leading (ring_dims > 0) or trailing
(ring_dims < 0) mesh dimensions to build rings for. If unspecified, build
rings for all but the first dimension.
</td>
</tr><tr>
<td>
`ring_axes`<a id="ring_axes"></a>
</td>
<td>
Optional; A permutation of ["x", "y", "z", "core"], specifying
the order of TPU topology axes to build rings in. If unspecified, default
to ["core", "x", "y", "z"].
</td>
</tr><tr>
<td>
`ring_bounds`<a id="ring_bounds"></a>
</td>
<td>
Optional; The maximum number of devices on each axis, in the x,
y, z, core order. If unspecified, default to physical topology limits.
</td>
</tr><tr>
<td>
`can_split_host_across_rings`<a id="can_split_host_across_rings"></a>
</td>
<td>
Optional; If true, devices attached to the same
host (i.e., DTensor client) may get assigned to different rings. Setting
it to false may cause some combinations of arguments to be infeasible; see
DeviceAssignmentTest.testCreateMesh[No]SplittingHosts* for examples.
</td>
</tr><tr>
<td>
`build_ring_across_rings`<a id="build_ring_across_rings"></a>
</td>
<td>
Optional; If true, also build a data-parallel ring
across model-parallel rings. This ring could be strided.
</td>
</tr><tr>
<td>
`rotate_ring_across_rings`<a id="rotate_ring_across_rings"></a>
</td>
<td>
Optional; If true, build the data-parallel ring in
column-major instead of row-major order.
</td>
</tr><tr>
<td>
`use_xla_spmd`<a id="use_xla_spmd"></a>
</td>
<td>
Boolean when True, will use XLA SPMD instead of
DTensor SPMD.
</td>
</tr>
</table>

