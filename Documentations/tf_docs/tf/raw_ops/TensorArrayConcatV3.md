description: Concat the elements from the TensorArray into value value.
robots: noindex

# tf.raw_ops.TensorArrayConcatV3

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Concat the elements from the TensorArray into value `value`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorArrayConcatV3`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorArrayConcatV3(
    handle, flow_in, dtype, element_shape_except0=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Takes `T` elements of shapes

  ```
  (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
  ```

and concatenates them into a Tensor of shape:

  ```
  (n0 + n1 + ... + n(T-1) x d0 x d1 x ...)
  ```

All elements must have the same shape (excepting the first dimension).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`handle`<a id="handle"></a>
</td>
<td>
A `Tensor` of type `resource`. The handle to a TensorArray.
</td>
</tr><tr>
<td>
`flow_in`<a id="flow_in"></a>
</td>
<td>
A `Tensor` of type `float32`.
A float scalar that enforces proper chaining of operations.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>. The type of the elem that is returned.
</td>
</tr><tr>
<td>
`element_shape_except0`<a id="element_shape_except0"></a>
</td>
<td>
An optional <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`. Defaults to `None`.
The expected shape of an element, if known,
excluding the first dimension. Used to validate the shapes of
TensorArray elements. If this shape is not fully specified, concatenating
zero-size TensorArrays is an error.
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
A tuple of `Tensor` objects (value, lengths).
</td>
</tr>
<tr>
<td>
`value`<a id="value"></a>
</td>
<td>
A `Tensor` of type `dtype`.
</td>
</tr><tr>
<td>
`lengths`<a id="lengths"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr>
</table>

