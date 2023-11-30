description: Computes offsets of concat inputs within its output.
robots: noindex

# tf.raw_ops.ConcatOffset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes offsets of concat inputs within its output.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ConcatOffset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ConcatOffset(
    concat_dim, shape, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### For example:



```
>>> x = [2, 2, 7]
>>> y = [2, 3, 7]
>>> z = [2, 9, 7]
>>> offsets = concat_offset(1, [x, y, z])
>>> [list(off.numpy()) for off in offsets]
[[0, 0, 0], [0, 2, 0], [0, 5, 0]]
```

This is typically used by gradient computations for a concat operation.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`concat_dim`<a id="concat_dim"></a>
</td>
<td>
A `Tensor` of type `int32`.
The dimension along which to concatenate.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A list of at least 2 `Tensor` objects with the same type in: `int32`, `int64`.
The `N` int32 or int64 vectors representing shape of tensors being concatenated.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list with the same length as `shape` of `Tensor` objects with the same type as `shape`.
</td>
</tr>

</table>

