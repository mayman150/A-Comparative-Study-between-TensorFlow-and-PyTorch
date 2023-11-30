description: Unpacks a DTensor into <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> components.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.unpack" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.unpack

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Unpacks a DTensor into <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> components.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.unpack(
    tensor: Any
) -> Sequence[Any]
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

See the documentation for `pack` for more information about how packing and
unpacking works.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
The DTensor to unpack.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The individual component tensors of the DTensor. This will include only the
client-local components, i.e. the components placed on the local devices.
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
When `unpack` is not called eagerly.
</td>
</tr>
</table>

