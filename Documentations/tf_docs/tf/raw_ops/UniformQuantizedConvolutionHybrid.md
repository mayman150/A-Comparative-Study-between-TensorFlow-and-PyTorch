description: Perform hybrid quantized convolution of float Tensor lhs and quantized Tensor rhs.
robots: noindex

# tf.raw_ops.UniformQuantizedConvolutionHybrid

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform hybrid quantized convolution of float Tensor `lhs` and quantized Tensor `rhs`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.UniformQuantizedConvolutionHybrid`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.UniformQuantizedConvolutionHybrid(
    lhs,
    rhs,
    rhs_scales,
    rhs_zero_points,
    Tout,
    padding,
    rhs_quantization_min_val,
    rhs_quantization_max_val,
    window_strides=[],
    explicit_padding=[],
    lhs_dilation=[],
    rhs_dilation=[],
    batch_group_count=1,
    feature_group_count=1,
    dimension_numbers=&#x27;&#x27;,
    rhs_quantization_axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given float `lhs` and quantized `rhs`, internally performs quantization on `lhs`,
and then performs quantized convolution on quantized `lhs` and `rhs`.

The internal quantization on `lhs` is a quantization to `Trhs`, dynamic range,
per-batch (per-axis along axis `dimension_numbers.input_batch_dimension`), asymmetric,
and not narrow range (the range is [Trhs_MIN, Trhs_MAX]).

`lhs` and `rhs` must be Tensors of same rank, and meet following shape conditions.
- lhs_feature % feature_group_count == 0
- lhs_feature % rhs_input_feature == 0
- lhs_feature / feature_group_count == rhs_input_feature
- rhs_output_feature % feature_group_count == 0
- lhs_batch % batch_group_count == 0
- rhs_output_feature % batch_group_count == 0

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
Must be a non-quantized Tensor of `Tlhs`, rank >= 3.
</td>
</tr><tr>
<td>
`rhs`<a id="rhs"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`.
Must be a quantized Tensor of `Trhs`, same rank as `lhs`.
</td>
</tr><tr>
<td>
`rhs_scales`<a id="rhs_scales"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value(s) used as scale factors when quantizing the original data that `rhs` represents.
Must be a scalar Tensor for per-tensor quantization,
or 1D Tensor of size `rhs.dim_size(kernel_output_feature_dimension)`, for per-channel quantization.
</td>
</tr><tr>
<td>
`rhs_zero_points`<a id="rhs_zero_points"></a>
</td>
<td>
A `Tensor` of type `int32`.
The int32 value(s) used as zero_point when quantizing original data that `rhs` represents.
Same shape condition as `rhs_scales`.
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
`padding`<a id="padding"></a>
</td>
<td>
A `string`.
string from: `"SAME"`, `"VALID"`, or `"EXPLICIT"`, indicating the type of padding algorithm to use.
</td>
</tr><tr>
<td>
`rhs_quantization_min_val`<a id="rhs_quantization_min_val"></a>
</td>
<td>
An `int`.
The min value of the quantized data stored in `rhs`.
For example, if `Trhs` is qint8, this must be set to -127 if narrow range quantized or -128 if not.
</td>
</tr><tr>
<td>
`rhs_quantization_max_val`<a id="rhs_quantization_max_val"></a>
</td>
<td>
An `int`.
The max value of the quantized data stored in `rhs`.
For example, if `Trhs` is qint8, this must be set to 127.
</td>
</tr><tr>
<td>
`window_strides`<a id="window_strides"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
The stride of the sliding window for each spatial dimension of `lhs`.
Must be an empty list (default) or a list of size (number of spatial dimensions).
If an empty list is provided, the stride for each spatial dimension is set to 1.
</td>
</tr><tr>
<td>
`explicit_padding`<a id="explicit_padding"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
If `padding` Attr is `"EXPLICIT"`, must be set as a list indicating
the explicit paddings at the start and end of each lhs spatial dimension.
Otherwise, this Attr is must be empty.

(If used,) Must be a list of size 2 * (number of lhs spatial dimensions),
where (explicit_padding[2 * i], explicit_padding[2 * i + 1]) indicates
spatial_dimensions[i] (start_padding, end_padding).
</td>
</tr><tr>
<td>
`lhs_dilation`<a id="lhs_dilation"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
The dilation factor to apply in each spatial dimension of `lhs`.
Must be an empty list (default) or a list of size (number of lhs spatial dimensions).
If empty list, the dilation for each lhs spatial dimension is set to 1.
</td>
</tr><tr>
<td>
`rhs_dilation`<a id="rhs_dilation"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
The dilation factor to apply in each spatial dimension of `rhs`.
Must be an empty list (default) or a list of size (number of rhs spatial dimensions).
If empty list, the dilation for each rhs spatial dimension is set to 1.
</td>
</tr><tr>
<td>
`batch_group_count`<a id="batch_group_count"></a>
</td>
<td>
An optional `int`. Defaults to `1`.
The number of batch groups. Used for grouped filters.
Must be a divisor of output_feature.
</td>
</tr><tr>
<td>
`feature_group_count`<a id="feature_group_count"></a>
</td>
<td>
An optional `int`. Defaults to `1`.
The number of feature groups. Used for grouped convolutions.
Must be a divisor of both lhs_feature and output_feature.
</td>
</tr><tr>
<td>
`dimension_numbers`<a id="dimension_numbers"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
Structure of dimension information for the convolution op.
Must be an empty string (default) or a serialized string of tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr proto.
If empty string, the default is `("NCHW", "OIHW", "NCHW")` (for a 2D convolution).
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
or per-channel quantization along kernel_output_feature_dimension is supported.
Thus, this attribute must be set to -1 or `dimension_numbers.kernel_output_feature_dimension`.
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
A `Tensor` of type `Tout`.
</td>
</tr>

</table>

