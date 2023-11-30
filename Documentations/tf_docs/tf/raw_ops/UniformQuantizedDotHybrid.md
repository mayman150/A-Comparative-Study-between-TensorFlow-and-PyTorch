description: Perform hybrid quantized dot of float Tensor lhs and quantized Tensor rhs.
robots: noindex

# tf.raw_ops.UniformQuantizedDotHybrid

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform hybrid quantized dot of float Tensor `lhs` and quantized Tensor `rhs`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformQuantizedDotHybrid`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformQuantizedDotHybrid(
    lhs,
    rhs,
    rhs_scales,
    rhs_zero_points,
    Tout,
    rhs_quantization_min_val,
    rhs_quantization_max_val,
    rhs_quantization_axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given float `lhs` and quantized `rhs`, internally performs quantization on `lhs`, and then performs quantized dot on quantized lhs and `rhs`.
The internal quantization on `lhs` is a quantization to qint8, dynamic range, per-batch (per-axis along axis 0), asymmetric, and not narrow range (the range is [-128, 127]).
`lhs` and `rhs` must be 2D Tensors and the lhs.dim_size(1) must match rhs.dim_size(0).
`rhs` must be quantized Tensor, where its data value is quantized using the formula:
quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`lhs`<a id="lhs"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`.
Must be a 2D Tensor of Tlhs.
</td>
</tr><tr>
<td>
`rhs`<a id="rhs"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`.
Must be a 2D Tensor of Trhs.
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
`Tout`<a id="Tout"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: <a href="../../tf.md#float32"><code>tf.float32</code></a>. The type of output Tensor.
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

