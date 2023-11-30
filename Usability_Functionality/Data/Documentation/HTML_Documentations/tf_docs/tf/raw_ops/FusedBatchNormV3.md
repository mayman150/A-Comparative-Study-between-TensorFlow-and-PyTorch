description: Batch normalization.
robots: noindex

# tf.raw_ops.FusedBatchNormV3

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Batch normalization.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.FusedBatchNormV3`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.FusedBatchNormV3(
    x,
    scale,
    offset,
    mean,
    variance,
    epsilon=0.0001,
    exponential_avg_factor=1,
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
`x`<a id="x"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
A 4D Tensor for input data.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `float32`.
A 1D Tensor for scaling factor, to scale the normalized x.
</td>
</tr><tr>
<td>
`offset`<a id="offset"></a>
</td>
<td>
A `Tensor`. Must have the same type as `scale`.
A 1D Tensor for offset, to shift to the normalized x.
</td>
</tr><tr>
<td>
`mean`<a id="mean"></a>
</td>
<td>
A `Tensor`. Must have the same type as `scale`.
A 1D Tensor for population mean. Used for inference only;
must be empty for training.
</td>
</tr><tr>
<td>
`variance`<a id="variance"></a>
</td>
<td>
A `Tensor`. Must have the same type as `scale`.
A 1D Tensor for population variance. Used for inference only;
must be empty for training.
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
`exponential_avg_factor`<a id="exponential_avg_factor"></a>
</td>
<td>
An optional `float`. Defaults to `1`.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
An optional `string` from: `"NHWC", "NCHW", "NDHWC", "NCDHW"`. Defaults to `"NHWC"`.
The data format for x and y. Either "NHWC" (default) or "NCHW".
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
A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2, reserve_space_3).
</td>
</tr>
<tr>
<td>
`y`<a id="y"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`batch_mean`<a id="batch_mean"></a>
</td>
<td>
A `Tensor`. Has the same type as `scale`.
</td>
</tr><tr>
<td>
`batch_variance`<a id="batch_variance"></a>
</td>
<td>
A `Tensor`. Has the same type as `scale`.
</td>
</tr><tr>
<td>
`reserve_space_1`<a id="reserve_space_1"></a>
</td>
<td>
A `Tensor`. Has the same type as `scale`.
</td>
</tr><tr>
<td>
`reserve_space_2`<a id="reserve_space_2"></a>
</td>
<td>
A `Tensor`. Has the same type as `scale`.
</td>
</tr><tr>
<td>
`reserve_space_3`<a id="reserve_space_3"></a>
</td>
<td>
A `Tensor`. Has the same type as `scale`.
</td>
</tr>
</table>

