description: Copies a tf.Tensor onto the DTensor device with the given layout.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.copy_to_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.copy_to_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Copies a tf.Tensor onto the DTensor device with the given layout.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.copy_to_mesh(
    tensor: Any,
    layout: <a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>,
    source_layout: Optional[<a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>] = None
) -> <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Copies a regular tf.Tensor onto the DTensor device. Use the mesh attached to
`layout` as target mesh. This method currently only supports replicated
layouts, or one-to-one copies for sharded layouts.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
A regular tf.Tensor to be copied as a DTensor.
</td>
</tr><tr>
<td>
`layout`<a id="layout"></a>
</td>
<td>
Target layout (and mesh) for the result DTensor.
</td>
</tr><tr>
<td>
`source_layout`<a id="source_layout"></a>
</td>
<td>
Source layout of the tensor before copy. This argument
is deprecated.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A DTensor on the DTensor device with the given layout.
</td>
</tr>

</table>

