description: Represents a Mesh configuration over a certain list of Mesh Dimensions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.Mesh" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="as_proto"/>
<meta itemprop="property" content="contains_dim"/>
<meta itemprop="property" content="coords"/>
<meta itemprop="property" content="device_location"/>
<meta itemprop="property" content="device_type"/>
<meta itemprop="property" content="dim_size"/>
<meta itemprop="property" content="from_device"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="from_string"/>
<meta itemprop="property" content="global_device_ids"/>
<meta itemprop="property" content="global_devices"/>
<meta itemprop="property" content="host_mesh"/>
<meta itemprop="property" content="is_remote"/>
<meta itemprop="property" content="is_single_device"/>
<meta itemprop="property" content="local_device_ids"/>
<meta itemprop="property" content="local_device_locations"/>
<meta itemprop="property" content="local_devices"/>
<meta itemprop="property" content="min_global_device_id"/>
<meta itemprop="property" content="num_local_devices"/>
<meta itemprop="property" content="shape"/>
<meta itemprop="property" content="to_string"/>
<meta itemprop="property" content="unravel_index"/>
<meta itemprop="property" content="use_xla_spmd"/>
</div>

# tf.experimental.dtensor.Mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>



Represents a Mesh configuration over a certain list of Mesh Dimensions.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.Mesh(
    dim_names: List[str],
    global_device_ids: np.ndarray,
    local_device_ids: List[int],
    local_devices: List[Union[tf_device.DeviceSpec, str]],
    mesh_name: str = &#x27;&#x27;,
    global_devices: Optional[List[Union[tf_device.DeviceSpec, str]]] = None,
    use_xla_spmd: bool = USE_XLA_SPMD
)
</code></pre>



<!-- Placeholder for "Used in" -->

A mesh consists of named dimensions with sizes, which describe how a set of
devices are arranged. Defining tensor layouts in terms of mesh dimensions
allows us to efficiently determine the communication required when computing
an operation with tensors of different layouts.

A mesh provides information not only about the placement of the tensors but
also the topology of the underlying devices. For example, we can group 8 TPUs
as a 1-D array for data parallelism or a `2x4` grid for (2-way) data
parallelism and (4-way) model parallelism.

Refer to [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
for in depth discussion and examples.

Note: the utilities <a href="../../../tf/experimental/dtensor/create_mesh.md"><code>dtensor.create_mesh</code></a> and
<a href="../../../tf/experimental/dtensor/create_distributed_mesh.md"><code>dtensor.create_distributed_mesh</code></a> provide a simpler API to create meshes for
single- or multi-client use cases.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dim_names`<a id="dim_names"></a>
</td>
<td>
A list of strings indicating dimension names.
</td>
</tr><tr>
<td>
`global_device_ids`<a id="global_device_ids"></a>
</td>
<td>
An ndarray of global device IDs is used to compose
DeviceSpecs describing the mesh. The shape of this array determines the
size of each mesh dimension. Values in this array should increment
sequentially from 0. This argument is the same for every DTensor client.
</td>
</tr><tr>
<td>
`local_device_ids`<a id="local_device_ids"></a>
</td>
<td>
A list of local device IDs equal to a subset of values
in global_device_ids. They indicate the position of local devices in the
global mesh. Different DTensor clients must contain distinct
local_device_ids contents. All local_device_ids from all DTensor clients
must cover every element in global_device_ids.
</td>
</tr><tr>
<td>
`local_devices`<a id="local_devices"></a>
</td>
<td>
The list of devices hosted locally. The elements correspond
1:1 to those of local_device_ids.
</td>
</tr><tr>
<td>
`mesh_name`<a id="mesh_name"></a>
</td>
<td>
The name of the mesh. Currently, this is rarely used, and is
mostly used to indicate whether it is a CPU, GPU, or TPU-based mesh.
</td>
</tr><tr>
<td>
`global_devices`<a id="global_devices"></a>
</td>
<td>
`optional`

The list of global devices. Set when multiple
  device meshes are in use.
</td>
</tr><tr>
<td>
`use_xla_spmd`<a id="use_xla_spmd"></a>
</td>
<td>
`optional`

Boolean when True, will use XLA SPMD instead of
  DTensor SPMD.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dim_names`<a id="dim_names"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`single_device`<a id="single_device"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`size`<a id="size"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
Returns the strides tensor array for this mesh.

If the mesh shape is `[a, b, c, d]`, then the strides array can be computed
as `[b*c*d, c*d, d, 1]`. This array can be useful in computing local device
offsets given a device ID. Using the same example, the device coordinates of
the mesh can be computed as:

```
[(device_id / (b*c*d)) % a,
 (device_id / (c*d))   % b,
 (device_id / (d))     % c,
 (device_id)           % d]
```

This is the same as `(device_id // mesh.strides) % mesh.shape`.
</td>
</tr>
</table>



## Methods

<h3 id="as_proto"><code>as_proto</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_proto()
</code></pre>

as_proto(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> tensorflow::dtensor::MeshProto


Returns the MeshProto protobuf message.

<h3 id="contains_dim"><code>contains_dim</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>contains_dim()
</code></pre>

contains_dim(self: tensorflow.python._pywrap_dtensor_device.Mesh, dim_name: str) -> bool


Returns True if a Mesh contains the given dimension name.

<h3 id="coords"><code>coords</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>coords(
    device_idx: int
) -> <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>
</code></pre>

Converts the device index into a tensor of mesh coordinates.


<h3 id="device_location"><code>device_location</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>device_location()
</code></pre>

device_location(self: tensorflow.python._pywrap_dtensor_device.Mesh, arg0: int) -> List[int]


<h3 id="device_type"><code>device_type</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>device_type()
</code></pre>

device_type(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> str


Returns the device_type of a Mesh.

<h3 id="dim_size"><code>dim_size</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dim_size()
</code></pre>

dim_size(self: tensorflow.python._pywrap_dtensor_device.Mesh, dim_name: str) -> int


Returns the size of mesh dimension.

<h3 id="from_device"><code>from_device</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_device(
    device: str
) -> 'Mesh'
</code></pre>

Constructs a single device mesh from a device string.


<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_proto(
    proto: layout_pb2.MeshProto
) -> 'Mesh'
</code></pre>

Construct a mesh instance from input `proto`.


<h3 id="from_string"><code>from_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_string(
    mesh_str: str
) -> 'Mesh'
</code></pre>




<h3 id="global_device_ids"><code>global_device_ids</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>global_device_ids() -> np.ndarray
</code></pre>

Returns a global device list as an array.


<h3 id="global_devices"><code>global_devices</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>global_devices()
</code></pre>

global_devices(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> List[str]


Returns a list of global device specs represented as strings.

<h3 id="host_mesh"><code>host_mesh</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>host_mesh() -> 'Mesh'
</code></pre>

Returns a host mesh.


<h3 id="is_remote"><code>is_remote</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_remote()
</code></pre>

is_remote(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> bool


Returns True if a Mesh contains only remote devices.

<h3 id="is_single_device"><code>is_single_device</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_single_device()
</code></pre>

is_single_device(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> bool


Returns True if the mesh represents a non-distributed device.

<h3 id="local_device_ids"><code>local_device_ids</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>local_device_ids()
</code></pre>

local_device_ids(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> Span[int]


Returns a list of local device IDs.

<h3 id="local_device_locations"><code>local_device_locations</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>local_device_locations() -> List[Dict[str, int]]
</code></pre>

Returns a list of local device locations.

A device location is a dictionary from dimension names to indices on those
dimensions.

<h3 id="local_devices"><code>local_devices</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>local_devices()
</code></pre>

local_devices(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> Span[str]


Returns a list of local device specs represented as strings.

<h3 id="min_global_device_id"><code>min_global_device_id</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>min_global_device_id()
</code></pre>

min_global_device_id(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> int


Returns the minimum global device ID.

<h3 id="num_local_devices"><code>num_local_devices</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_local_devices()
</code></pre>

num_local_devices(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> int


Returns the number of local devices.

<h3 id="shape"><code>shape</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>shape()
</code></pre>

shape(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> List[int]


Returns the shape of the mesh.

<h3 id="to_string"><code>to_string</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_string()
</code></pre>

to_string(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> str


Returns string representation of Mesh.

<h3 id="unravel_index"><code>unravel_index</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unravel_index()
</code></pre>

Returns a dictionary from device ID to {dim_name: dim_index}.

For example, for a 3x2 mesh, return this:

```
  { 0: {'x': 0, 'y', 0},
    1: {'x': 0, 'y', 1},
    2: {'x': 1, 'y', 0},
    3: {'x': 1, 'y', 1},
    4: {'x': 2, 'y', 0},
    5: {'x': 2, 'y', 1} }
```

<h3 id="use_xla_spmd"><code>use_xla_spmd</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>use_xla_spmd()
</code></pre>

use_xla_spmd(self: tensorflow.python._pywrap_dtensor_device.Mesh) -> bool


Returns True if Mesh will use XLA for SPMD instead of DTensor SPMD.

<h3 id="__contains__"><code>__contains__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__()
</code></pre>

__contains__(self: tensorflow.python._pywrap_dtensor_device.Mesh, dim_name: str) -> bool


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__()
</code></pre>

__eq__(self: tensorflow.python._pywrap_dtensor_device.Mesh, arg0: tensorflow.python._pywrap_dtensor_device.Mesh) -> bool


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    dim_name: str
) -> MeshDimension
</code></pre>






