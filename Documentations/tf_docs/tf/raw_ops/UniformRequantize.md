description: Given quantized tensor input, requantize it with new quantization parameters.
robots: noindex

# tf.raw_ops.UniformRequantize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Given quantized tensor `input`, requantize it with new quantization parameters.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformRequantize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformRequantize(
    input,
    input_scales,
    input_zero_points,
    output_scales,
    output_zero_points,
    Tout,
    input_quantization_min_val,
    input_quantization_max_val,
    output_quantization_min_val,
    output_quantization_max_val,
    input_quantization_axis=-1,
    output_quantization_axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given quantized tensor `input`, which was quantized using {input_scales, input_zero_points, input_quantization_axis, input_quantization_min_val, input_quantization_max_val},
requantize it to a tensor, which is quantized using {output_scales, output_zero_points, output_quantization_axis, output_quantization_min_val, output_quantization_max_val}.
The requantization is done by using the formula:
output_quantized_data = clip(
  (input_quantized_data - input_zero_point) * (input_scale / output_scale) + output_zero_point,
  output_quantization_min_val,
  output_quantization_max_val)

Per-tensor and per-axis quantization supported cases are followings:
* per-tensor -> per-tensor
* per-tensor -> per-axis
* per-axis -> per-axis where input_quantization_axis equals output_quantization_axis.
i.e. At least one among input_quantization_axis and output_quantization_axis must be -1, or two must be equal.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `qint32`.
Must be a Tensor of Tin.
</td>
</tr><tr>
<td>
`input_scales`<a id="input_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale(s) when quantizing original data that `input` represents.
Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
</td>
</tr><tr>
<td>
`input_zero_points`<a id="input_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero_point(s) when quantizing original data that `input` represents.
Same shape condition as scales.
</td>
</tr><tr>
<td>
`output_scales`<a id="output_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) to use as new scale(s) to quantize original data that `input` represents.
Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
</td>
</tr><tr>
<td>
`output_zero_points`<a id="output_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) to use as new zero_point(s) to quantize original data that `input` represents.
Same shape condition as scales.
</td>
</tr><tr>
<td>
`Tout`<a id="Tout"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.qint32`.
The type of output Tensor. A tf.DType from: tf.qint8, tf.qint32
</td>
</tr><tr>
<td>
`input_quantization_min_val`<a id="input_quantization_min_val"></a>
</td>
<td>
An `int`.
The quantization min value that was used when quantizing original data that `input` represents.
The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
`(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`input_quantization_max_val`<a id="input_quantization_max_val"></a>
</td>
<td>
An `int`.
The quantization max value that was used when quantizing original data that `input` represents.
The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
`(Tout max)` for both narrow range and not narrow range.
For example, if Tin is qint8, this is set to 127.
</td>
</tr><tr>
<td>
`output_quantization_min_val`<a id="output_quantization_min_val"></a>
</td>
<td>
An `int`.
The new quantization min value to quantize original data that `input` represents.
</td>
</tr><tr>
<td>
`output_quantization_max_val`<a id="output_quantization_max_val"></a>
</td>
<td>
An `int`.
The new quantization max value to quantize original data that `input` represents.
</td>
</tr><tr>
<td>
`input_quantization_axis`<a id="input_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
The quantization axis that was used when quantizing original data that `input` represents.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
</td>
</tr><tr>
<td>
`output_quantization_axis`<a id="output_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
The new quantization axis to use to quantize original data that `input` represents.
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
A `Tensor` of type `Tout`.
</td>
</tr>

</table>

