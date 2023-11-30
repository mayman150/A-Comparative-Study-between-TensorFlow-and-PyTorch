description: Computes the gradients of convolution with respect to the filter.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.conv2d_backprop_filter_v2" />
<meta itemprop="path" content="Stable" />
</div>

# tf.conv2d_backprop_filter_v2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the gradients of convolution with respect to the filter.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.conv2d_backprop_filter_v2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.conv2d_backprop_filter_v2(
    input: Annotated[Any, TV_Conv2DBackpropFilterV2_T],
    filter: Annotated[Any, TV_Conv2DBackpropFilterV2_T],
    out_backprop: Annotated[Any, TV_Conv2DBackpropFilterV2_T],
    strides,
    padding: str,
    use_cudnn_on_gpu: bool = True,
    explicit_paddings=[],
    data_format: str = &#x27;NHWC&#x27;,
    dilations=[1, 1, 1, 1],
    name=None
) -> Annotated[Any, TV_Conv2DBackpropFilterV2_T]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
4-D with shape `[batch, in_height, in_width, in_channels]`.
</td>
</tr><tr>
<td>
`filter`<a id="filter"></a>
</td>
<td>
A `Tensor`. Must have the same type as `input`.
4-D with shape `[filter_height, filter_width, in_channels, out_channels]`.
Only shape of tensor is used.
</td>
</tr><tr>
<td>
`out_backprop`<a id="out_backprop"></a>
</td>
<td>
A `Tensor`. Must have the same type as `input`.
4-D with shape `[batch, out_height, out_width, out_channels]`.
Gradients w.r.t. the output of the convolution.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
A list of `ints`.
The stride of the sliding window for each dimension of the input
of the convolution. Must be in the same order as the dimension specified with
format.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
A `string` from: `"SAME", "VALID", "EXPLICIT"`.
The type of padding algorithm to use.
</td>
</tr><tr>
<td>
`use_cudnn_on_gpu`<a id="use_cudnn_on_gpu"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
</td>
</tr><tr>
<td>
`explicit_paddings`<a id="explicit_paddings"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
If `padding` is `"EXPLICIT"`, the list of explicit padding amounts. For the ith
dimension, the amount of padding inserted before and after the dimension is
`explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
`padding` is not `"EXPLICIT"`, `explicit_paddings` must be empty.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
Specify the data format of the input and output data. With the
default format "NHWC", the data is stored in the order of:
    [batch, in_height, in_width, in_channels].
Alternatively, the format could be "NCHW", the data storage order of:
    [batch, in_channels, in_height, in_width].
</td>
</tr><tr>
<td>
`dilations`<a id="dilations"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
1-D tensor of length 4.  The dilation factor for each dimension of
`input`. If set to k > 1, there will be k-1 skipped cells between each filter
element on that dimension. The dimension order is determined by the value of
`data_format`, see above for details. Dilations in the batch and depth
dimensions must be 1.
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
A `Tensor`. Has the same type as `input`.
</td>
</tr>

</table>

