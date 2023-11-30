description: Creates a distributed mesh.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.create_distributed_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.create_distributed_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/mesh_util.py">View source</a>



Creates a distributed mesh.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.create_distributed_mesh(
    mesh_dims: Union[List[Tuple[str, int]], Dict[str, int]],
    mesh_name: str = &#x27;&#x27;,
    local_devices: Optional[List[Union[tf_device.DeviceSpec, str]]] = None,
    device_type: Optional[str] = None,
    use_xla_spmd: bool = layout.USE_XLA_SPMD
) -> <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This is similar to `create_mesh`, but with a different set of arguments to
create a mesh that spans evenly across a multi-client DTensor cluster.

For CPU and GPU meshes, users can choose to use fewer local devices than what
is available `local_devices`.

For TPU, only meshes that uses all TPU cores is supported by the DTensor
runtime.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh_dims`<a id="mesh_dims"></a>
</td>
<td>
A dict of dim_name: dim_size, or a list of (dim_name, dim_size)
tuples. e.g. `{'x' : 4, 'y' : 1}` or `[('x', 4), ('y', 1)]`.
</td>
</tr><tr>
<td>
`mesh_name`<a id="mesh_name"></a>
</td>
<td>
Name of the created mesh. Defaults to ''.
</td>
</tr><tr>
<td>
`local_devices`<a id="local_devices"></a>
</td>
<td>
String representations of devices to use. This is the device
part of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available local
logical devices.
</td>
</tr><tr>
<td>
`device_type`<a id="device_type"></a>
</td>
<td>
Type of device to build the mesh for. Defaults to 'CPU'.
Supported values are 'CPU', 'GPU', 'TPU'.6
</td>
</tr><tr>
<td>
`use_xla_spmd`<a id="use_xla_spmd"></a>
</td>
<td>
Boolean when True, will use XLA SPMD instead of DTensor SPMD.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A mesh that spans evenly across all DTensor clients in the cluster.
</td>
</tr>

</table>

