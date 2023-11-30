description: Perform quantized add of quantized Tensor lhs and quantized Tensor rhs to make quantized output.
robots: noindex

# tf.raw_ops.UniformQuantizedAdd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform quantized add of quantized Tensor `lhs` and quantized Tensor `rhs` to make quantized `output`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformQuantizedAdd`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformQuantizedAdd(
    lhs,
    rhs,
    lhs_scales,
    lhs_zero_points,
    rhs_scales,
    rhs_zero_points,
    output_scales,
    output_zero_points,
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

Given quantized `lhs` and quantized `rhs`, performs quantized add on `lhs` and `rhs` to make quantized `output`.

`UniformQuantizedAdd` follows Numpy broadcasting rules.
The two input array shapes are compared element-wise.
Starting with the trailing dimensions, the two dimensions either have to be equal or one of them needs to be 1.

`lhs` and `rhs` must be quantized Tensor, where data value is quantized using the formula:
```
quantized_data = clip(original_data / scale + zero_point, quantization_min_val, quantization_max_val)
```
`output` is also quantized, using the same formula.

If `lhs` and `output` is both per-axis quantized, the quantization axis must match.
Also, if `rhs` and `output` is both per-axis quantized, the quantization axis must match.
*Match* means the axis must match when adding, regarding the broadcasting.
i.e. For both operands `lhs` and `rhs`,
if `operand.quantization_axis` >= 0 and `output.quantization_axis` >= 0,
`operand.dims` - `operand.quantization_axis` must be equal to `output.dims` - `output.quantization_axis`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`lhs`<a id="lhs"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint32`.
Must be a quantized tensor.
</td>
</tr><tr>
<td>
`rhs`<a id="rhs"></a>
</td>
<td>
A `Tensor`. Must have the same type as `lhs`.
Must be a quantized tensor.
</td>
</tr><tr>
<td>
`lhs_scales`<a id="lhs_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale factors when quantizing the original data that `lhs` represents.
</td>
</tr><tr>
<td>
`lhs_zero_points`<a id="lhs_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero points when quantizing original data that `lhs` represents.
Must have same shape with `lhs_scales`.
</td>
</tr><tr>
<td>
`rhs_scales`<a id="rhs_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
</td>
</tr><tr>
<td>
`rhs_zero_points`<a id="rhs_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero points when quantizing original data that `rhs` represents.
Must have same shape with `rhs_scales`.
</td>
</tr><tr>
<td>
`output_scales`<a id="output_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) to use as scale factors when quantizing original data that `output` represents.
</td>
</tr><tr>
<td>
`output_zero_points`<a id="output_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero points when quantizing original data that output represents.
Must have same shape with `output_scales`.
</td>
</tr><tr>
<td>
`lhs_quantization_min_val`<a id="lhs_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in `lhs`.
For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`lhs_quantization_max_val`<a id="lhs_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in `lhs`.
For example, if `Tin` is `qint8`, this must be set to 127.
</td>
</tr><tr>
<td>
`rhs_quantization_min_val`<a id="rhs_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in `rhs`.
For example, if `Tin` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`rhs_quantization_max_val`<a id="rhs_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in `rhs`.
For example, if `Tin` is `qint8`, this must be set to 127.
</td>
</tr><tr>
<td>
`output_quantization_min_val`<a id="output_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in `output`.
For example, if  `Tout` is `qint8`, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`output_quantization_max_val`<a id="output_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in `output`.
For example, if `Tout` is `qint8`, this must be set to 127.
</td>
</tr><tr>
<td>
`lhs_quantization_axis`<a id="lhs_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization.
For the `lhs`, only per-tensor quantization is supported.
Thus, this must be set to -1.
Other values will raise error at OpKernel construction.
</td>
</tr><tr>
<td>
`rhs_quantization_axis`<a id="rhs_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization.
For the `rhs`, only per-tensor quantization
or per-channel quantization along `kernel_output_feature_dimension` is supported.
Thus, this must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
Other values will raise error at OpKernel construction.
</td>
</tr><tr>
<td>
`output_quantization_axis`<a id="output_quantization_axis"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Indicates the dimension index of the tensor where per-axis quantization is applied for the slices along that dimension.
If set to -1 (default), this indicates per-tensor quantization.
For the `output`, only per-tensor quantization or per-channel quantization along `output_feature_dimension` is supported.
Thus, this must be set to -1 or `dimension_numbers.output_feature_dimension`.
Other values will raise error at OpKernel construction.
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
A `Tensor`. Has the same type as `lhs`.
</td>
</tr>

</table>

