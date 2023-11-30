description: Perform quantized dot of quantized Tensor lhs and quantized Tensor rhs to make quantized output.
robots: noindex

# tf.raw_ops.UniformQuantizedDot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform quantized dot of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformQuantizedDot`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformQuantizedDot(
    lhs,
    rhs,
    lhs_scales,
    lhs_zero_points,
    rhs_scales,
    rhs_zero_points,
    output_scales,
    output_zero_points,
    Tout,
    lhs_quantization_min_val,
    lhs_quantization_max_val,
    rhs_quantization_min_val,
    rhs_quantization_max_val,
    output_quantization_min_val,
    output_quantization_max_val,
    lhs_quantization_axis=-1,
    rhs_quantization_axis=-1,
    output_quantization_axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given quantized `lhs` and quantized `rhs`, performs quantized dot on `lhs` and `rhs` to make quantized `output`.
`lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
`lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).
`output` is also quantized, using the same formula.
If `rhs` is per-tensor quantized, `output` must be also per-tensor quantized.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`lhs`<a id="lhs"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`.
Must be a 2D Tensor of Tin.
</td>
</tr><tr>
<td>
`rhs`<a id="rhs"></a>
</td>
<td>
A `Tensor`. Must have the same type as `lhs`.
Must be a 2D Tensor of Tin.
</td>
</tr><tr>
<td>
`lhs_scales`<a id="lhs_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale when quantizing original data that lhs represents.
Must be a scalar Tensor (lhs supports only per-tensor quantization).
</td>
</tr><tr>
<td>
`lhs_zero_points`<a id="lhs_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero_point when quantizing original data that lhs represents.
Same shape condition as lhs_scales.
</td>
</tr><tr>
<td>
`rhs_scales`<a id="rhs_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale when quantizing original data that rhs represents.
Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (rhs.dim_size(1),) (per-channel quantization).
</td>
</tr><tr>
<td>
`rhs_zero_points`<a id="rhs_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero_point when quantizing original data that rhs represents.
Same shape condition as rhs_scales.
</td>
</tr><tr>
<td>
`output_scales`<a id="output_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) to use as scales when quantizing original data that output represents.
Must be a scalar Tensor (per-tensor quantization) or 1D Tensor of size (output.dim_size(1),) (per-channel quantization).
If rhs is per-tensor quantized, output must be also per-tensor quantized.
This means that if rhs_scales and rhs_zero_points are scalar Tensors, output_scales and output_zero_points must be scalar Tensors as well.
</td>
</tr><tr>
<td>
`output_zero_points`<a id="output_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero_point when quantizing original data that output represents.
Same shape condition as rhs_scales.
</td>
</tr><tr>
<td>
`Tout`<a id="Tout"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: <a href="../../tf.md#qint32"><code>tf.qint32</code></a>. The type of output Tensor.
</td>
</tr><tr>
<td>
`lhs_quantization_min_val`<a id="lhs_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in lhs.
For example, if Tin is qint8, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`lhs_quantization_max_val`<a id="lhs_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in rhs.
For example, if Tin is qint8, this must be set to 127.
</td>
</tr><tr>
<td>
`rhs_quantization_min_val`<a id="rhs_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in rhs.
For example, if Trhs is qint8, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`rhs_quantization_max_val`<a id="rhs_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in rhs.
For example, if Trhs is qint8, this must be set to 127.
</td>
</tr><tr>
<td>
`output_quantization_min_val`<a id="output_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in output.
For example, if Tout is qint8, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`output_quantization_max_val`<a id="output_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in output.
For example, if Tout is qint8, this must be set to 127.
</td>
</tr><tr>
<td>
`lhs_quantization_axis`<a id="lhs_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization.
For dot op lhs, only per-tensor quantization is supported.
Thus, this attribute must be set to -1. Other values are rejected.
</td>
</tr><tr>
<td>
`rhs_quantization_axis`<a id="rhs_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization.
For dot op rhs, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
Thus, this attribute must be set to -1 or 1. Other values are rejected.
</td>
</tr><tr>
<td>
`output_quantization_axis`<a id="output_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization.
For dot op output, only per-tensor quantization or per-channel quantization along dimension 1 is supported.
Thus, this attribute must be set to -1 or 1. Other values are rejected.
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

