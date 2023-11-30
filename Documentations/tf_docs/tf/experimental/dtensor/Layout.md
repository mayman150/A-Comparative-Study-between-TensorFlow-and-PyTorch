description: Represents the layout information of a DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.Layout" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="as_proto"/>
<meta itemprop="property" content="batch_sharded"/>
<meta itemprop="property" content="delete"/>
<meta itemprop="property" content="from_device"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="from_single_device_mesh"/>
<meta itemprop="property" content="from_string"/>
<meta itemprop="property" content="inner_sharded"/>
<meta itemprop="property" content="is_batch_parallel"/>
<meta itemprop="property" content="is_fully_replicated"/>
<meta itemprop="property" content="is_single_device"/>
<meta itemprop="property" content="num_shards"/>
<meta itemprop="property" content="offset_to_shard"/>
<meta itemprop="property" content="offset_tuple_to_global_index"/>
<meta itemprop="property" content="replicated"/>
<meta itemprop="property" content="to_parted"/>
<meta itemprop="property" content="to_string"/>
</div>

# tf.experimental.dtensor.Layout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>



Represents the layout information of a DTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.Layout(
    sharding_specs: List[str],
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

A layout describes how a distributed tensor is partitioned across a mesh (and
thus across devices). For each axis of the tensor, the corresponding
sharding spec indicates which dimension of the mesh it is sharded over. A
special sharding spec `UNSHARDED` indicates that axis is replicated on
all the devices of that mesh.

Refer to [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview)
for in depth discussion and examples.

For example, let's consider a 1-D mesh:

```
Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"], [("x", 6)])
```

This mesh arranges 6 TPU devices into a 1-D array. `Layout([UNSHARDED], mesh)`
is a layout for rank-1 tensor which is replicated on the 6 devices.

For another example, let's consider a 2-D mesh:

```
Mesh(["TPU:0", "TPU:1", "TPU:2", "TPU:3", "TPU:4", "TPU:5"],
     [("x", 3), ("y", 2)])
```

This mesh arranges 6 TPU devices into a `3x2` 2-D array.
`Layout(["x", UNSHARDED], mesh)` is a layout for rank-2 tensor whose first
axis is sharded on mesh dimension "x" and the second axis is replicated. If we
place `np.arange(6).reshape((3, 2))` using this layout, the individual
components tensors would look like:

```
Device  |  Component
 TPU:0     [[0, 1]]
 TPU:1     [[0, 1]]
 TPU:2     [[2, 3]]
 TPU:3     [[2, 3]]
 TPU:4     [[4, 5]]
 TPU:5     [[4, 5]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sharding_specs`<a id="sharding_specs"></a>
</td>
<td>
List of sharding specifications, each corresponding to a
tensor axis. Each specification (dim_sharding) can either be a mesh
dimension or the special value UNSHARDED.
</td>
</tr><tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>
A mesh configuration for the Tensor.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`rank`<a id="rank"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`sharding_specs`<a id="sharding_specs"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`type`<a id="type"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="as_proto"><code>as_proto</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_proto()
</code></pre>

as_proto(self: tensorflow.python._pywrap_dtensor_device.Layout) -> tensorflow::dtensor::LayoutProto


Returns the LayoutProto protobuf message.

<h3 id="batch_sharded"><code>batch_sharded</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>batch_sharded(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    batch_dim: str,
    rank: int,
    axis: int = 0
) -> 'Layout'
</code></pre>

Returns a layout sharded on batch dimension.


<h3 id="delete"><code>delete</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>delete(
    dims: List[int]
) -> 'Layout'
</code></pre>

Returns the layout with the give dimensions deleted.


<h3 id="from_device"><code>from_device</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_device(
    device: str
) -> 'Layout'
</code></pre>

Constructs a single device layout from a single device mesh.


<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_proto(
    layout_proto: layout_pb2.LayoutProto
) -> 'Layout'
</code></pre>

Creates an instance from a LayoutProto.


<h3 id="from_single_device_mesh"><code>from_single_device_mesh</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_single_device_mesh(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
) -> 'Layout'
</code></pre>

Constructs a single device layout from a single device mesh.


<h3 id="from_string"><code>from_string</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_string(
    layout_str: str
) -> 'Layout'
</code></pre>

Creates an instance from a human-readable string.


<h3 id="inner_sharded"><code>inner_sharded</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>inner_sharded(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    inner_dim: str,
    rank: int
) -> 'Layout'
</code></pre>

Returns a layout sharded on inner dimension.


<h3 id="is_batch_parallel"><code>is_batch_parallel</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_batch_parallel()
</code></pre>

is_batch_parallel(self: tensorflow.python._pywrap_dtensor_device.Layout) -> bool


<h3 id="is_fully_replicated"><code>is_fully_replicated</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_fully_replicated()
</code></pre>

is_fully_replicated(self: tensorflow.python._pywrap_dtensor_device.Layout) -> bool


Returns True if all tensor axes are replicated.

<h3 id="is_single_device"><code>is_single_device</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_single_device()
</code></pre>

is_single_device(self: tensorflow.python._pywrap_dtensor_device.Layout) -> bool


Returns True if the Layout represents a non-distributed device.

<h3 id="num_shards"><code>num_shards</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_shards()
</code></pre>

num_shards(self: tensorflow.python._pywrap_dtensor_device.Layout, idx: int) -> int


Returns the number of shards for tensor dimension `idx`.

<h3 id="offset_to_shard"><code>offset_to_shard</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>offset_to_shard()
</code></pre>

Mapping from offset in a flattened list to shard index.


<h3 id="offset_tuple_to_global_index"><code>offset_tuple_to_global_index</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>offset_tuple_to_global_index(
    offset_tuple
)
</code></pre>

Mapping from offset to index in global tensor.


<h3 id="replicated"><code>replicated</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>replicated(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    rank: int
) -> 'Layout'
</code></pre>

Returns a replicated layout of rank `rank`.


<h3 id="to_parted"><code>to_parted</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/layout.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_parted() -> 'Layout'
</code></pre>

Returns a "parted" layout from a static layout.

A parted layout contains axes that are treated as independent by most of
SPMD expanders.

FIXME(b/285905569): The exact semantics is still being investigated.

<h3 id="to_string"><code>to_string</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_string()
</code></pre>

to_string(self: tensorflow.python._pywrap_dtensor_device.Layout) -> str


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__()
</code></pre>

__eq__(self: tensorflow.python._pywrap_dtensor_device.Layout, arg0: tensorflow.python._pywrap_dtensor_device.Layout) -> bool




