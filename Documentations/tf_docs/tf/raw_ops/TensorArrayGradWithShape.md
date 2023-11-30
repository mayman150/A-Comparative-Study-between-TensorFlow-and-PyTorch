description: Creates a TensorArray for storing multiple gradients of values in the given handle.
robots: noindex

# tf.raw_ops.TensorArrayGradWithShape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a TensorArray for storing multiple gradients of values in the given handle.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorArrayGradWithShape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorArrayGradWithShape(
    handle, flow_in, shape_to_prepend, source, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Similar to TensorArrayGradV3. However it creates an accumulator with an
expanded shape compared to the input TensorArray whose gradient is being
computed. This enables multiple gradients for the same TensorArray to be
calculated using the same accumulator.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`handle`<a id="handle"></a>
</td>
<td>
A `Tensor` of type `resource`.
The handle to the forward TensorArray.
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
`shape_to_prepend`<a id="shape_to_prepend"></a>
</td>
<td>
A `Tensor` of type `int32`.
An int32 vector representing a shape. Elements in the gradient accumulator will
have shape which is this shape_to_prepend value concatenated with shape of the
elements in the TensorArray corresponding to the input handle.
</td>
</tr><tr>
<td>
`source`<a id="source"></a>
</td>
<td>
A `string`.
The gradient source string, used to decide which gradient TensorArray
to return.
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
A tuple of `Tensor` objects (grad_handle, flow_out).
</td>
</tr>
<tr>
<td>
`grad_handle`<a id="grad_handle"></a>
</td>
<td>
A `Tensor` of type `resource`.
</td>
</tr><tr>
<td>
`flow_out`<a id="flow_out"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

