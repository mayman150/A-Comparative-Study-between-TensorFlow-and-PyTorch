description: Perform quantization on Tensor input.
robots: noindex

# tf.raw_ops.UniformQuantize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform quantization on Tensor `input`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformQuantize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformQuantize(
    input,
    scales,
    zero_points,
    Tout,
    quantization_min_val,
    quantization_max_val,
    quantization_axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given `input`, `scales` and `zero_points`, performs quantization using the formula:
quantized_data = floor(input_data * (1.0f / scale) + 0.5f) + zero_point

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`.
Must be a Tensor of Tin.
</td>
</tr><tr>
<td>
`scales`<a id="scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) to use as scale(s) to quantize `input`.
Must be a scalar Tensor if quantization_axis is -1 (per-tensor quantization), otherwise 1D Tensor of size (input.dim_size(quantization_axis),) (per-axis quantization).
</td>
</tr><tr>
<td>
`zero_points`<a id="zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) to use as zero_point(s) to quantize `input`.
Same shape condition as scales.
</td>
</tr><tr>
<td>
`Tout`<a id="Tout"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.qint32`.
The type of output Tensor. A tf.DType from: tf.float32
</td>
</tr><tr>
<td>
`quantization_min_val`<a id="quantization_min_val"></a>
</td>
<td>
An `int`.
The quantization min value to quantize `input`.
The purpose of this attribute is typically (but not limited to) to indicate narrow range, where this is set to:
`(Tin lowest) + 1` if narrow range, and `(Tin lowest)` otherwise.
For example, if Tin is qint8, this is set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`quantization_max_val`<a id="quantization_max_val"></a>
</td>
<td>
An `int`.
The quantization max value to quantize `input`.
The purpose of this attribute is typically (but not limited to) indicate narrow range, where this is set to:
`(Tout max)` for both narrow range and not narrow range.
For example, if Tin is qint8, this is set to 127.
</td>
</tr><tr>
<td>
`quantization_axis`<a id="quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization. Otherwise, it must be set within range [0, input.dims()).
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

