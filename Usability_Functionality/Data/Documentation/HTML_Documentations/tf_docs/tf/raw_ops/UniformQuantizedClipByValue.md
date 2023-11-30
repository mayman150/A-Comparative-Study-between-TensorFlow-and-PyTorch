description: Perform clip by value on the quantized Tensor operand.
robots: noindex

# tf.raw_ops.UniformQuantizedClipByValue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform clip by value on the quantized Tensor `operand`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformQuantizedClipByValue`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformQuantizedClipByValue(
    operand,
    min,
    max,
    scales,
    zero_points,
    quantization_min_val,
    quantization_max_val,
    quantization_axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given quantized `operand` which was quantized using `scales` and `zero_points`, performs clip by value using `min` and `max` values.
If quantization_axis is -1 (per-tensor quantized), the entire operand is clipped using scalar min, max.
Otherwise (per-channel quantized), the clipping is also done per-channel.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`operand`<a id="operand"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint32`.
Must be a Tensor of T.
</td>
</tr><tr>
<td>
`min`<a id="min"></a>
</td>
<td>
A `Tensor`. Must have the same type as `operand`.
The min value(s) to clip operand. Must be a Tensor of T.
Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
</td>
</tr><tr>
<td>
`max`<a id="max"></a>
</td>
<td>
A `Tensor`. Must have the same type as `operand`.
The min value(s) to clip operand. Must be a Tensor of T.
Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
</td>
</tr><tr>
<td>
`scales`<a id="scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale(s) when quantizing `operand`, `min` and `max`.
Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (operand.dim_size(quantization_axis),) (per-axis quantization).
</td>
</tr><tr>
<td>
`zero_points`<a id="zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero_point(s) when quantizing `operand`, `min` and `max`.
Same shape condition as scales.
</td>
</tr><tr>
<td>
`quantization_min_val`<a id="quantization_min_val"></a>
</td>
<td>
An `int`.
The quantization min value that was used when operand was quantized.
</td>
</tr><tr>
<td>
`quantization_max_val`<a id="quantization_max_val"></a>
</td>
<td>
An `int`.
The quantization max value that was used when operand was quantized.
</td>
</tr><tr>
<td>
`quantization_axis`<a id="quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, operand.dims()).
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
A `Tensor`. Has the same type as `operand`.
</td>
</tr>

</table>

