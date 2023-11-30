description: Gradient for batch normalization.
robots: noindex

# tf.raw_ops.FusedBatchNormGrad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Gradient for batch normalization.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.FusedBatchNormGrad`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.FusedBatchNormGrad(
    y_backprop,
    x,
    scale,
    reserve_space_1,
    reserve_space_2,
    epsilon=0.0001,
    data_format=&#x27;NHWC&#x27;,
    is_training=True,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
The size of 1D Tensors matches the dimension C of the 4D Tensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`y_backprop`<a id="y_backprop"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`.
A 4D Tensor for the gradient with respect to y.
</td>
</tr><tr>
<td>
`x`<a id="x"></a>
</td>
<td>
A `Tensor`. Must have the same type as `y_backprop`.
A 4D Tensor for input data.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
A `Tensor`. Must have the same type as `y_backprop`.
A 1D Tensor for scaling factor, to scale the normalized x.
</td>
</tr><tr>
<td>
`reserve_space_1`<a id="reserve_space_1"></a>
</td>
<td>
A `Tensor`. Must have the same type as `y_backprop`.
When is_training is True, a 1D Tensor for the computed batch
mean to be reused in gradient computation. When is_training is
False, a 1D Tensor for the population mean to be reused in both
1st and 2nd order gradient computation.
</td>
</tr><tr>
<td>
`reserve_space_2`<a id="reserve_space_2"></a>
</td>
<td>
A `Tensor`. Must have the same type as `y_backprop`.
When is_training is True, a 1D Tensor for the computed batch
variance (inverted variance in the cuDNN case) to be reused in
gradient computation. When is_training is False, a 1D Tensor
for the population variance to be reused in both 1st and 2nd
order gradient computation.
</td>
</tr><tr>
<td>
`epsilon`<a id="epsilon"></a>
</td>
<td>
An optional `float`. Defaults to `0.0001`.
A small float number added to the variance of x.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
The data format for y_backprop, x, x_backprop.
Either "NHWC" (default) or "NCHW".
</td>
</tr><tr>
<td>
`is_training`<a id="is_training"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
A bool value to indicate the operation is for training (default)
or inference.
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
A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).
</td>
</tr>
<tr>
<td>
`x_backprop`<a id="x_backprop"></a>
</td>
<td>
A `Tensor`. Has the same type as `y_backprop`.
</td>
</tr><tr>
<td>
`scale_backprop`<a id="scale_backprop"></a>
</td>
<td>
A `Tensor`. Has the same type as `y_backprop`.
</td>
</tr><tr>
<td>
`offset_backprop`<a id="offset_backprop"></a>
</td>
<td>
A `Tensor`. Has the same type as `y_backprop`.
</td>
</tr><tr>
<td>
`reserve_space_3`<a id="reserve_space_3"></a>
</td>
<td>
A `Tensor`. Has the same type as `y_backprop`.
</td>
</tr><tr>
<td>
`reserve_space_4`<a id="reserve_space_4"></a>
</td>
<td>
A `Tensor`. Has the same type as `y_backprop`.
</td>
</tr>
</table>

