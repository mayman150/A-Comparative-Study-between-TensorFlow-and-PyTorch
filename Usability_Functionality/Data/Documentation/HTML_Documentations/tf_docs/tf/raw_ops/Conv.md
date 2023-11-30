description: Computes a N-D convolution given (N+1+batch_dims)-D input and (N+2)-D filter tensors.
robots: noindex

# tf.raw_ops.Conv

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes a N-D convolution given (N+1+batch_dims)-D `input` and (N+2)-D `filter` tensors.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.Conv`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.Conv(
    input,
    filter,
    strides,
    padding,
    explicit_paddings=[],
    data_format=&#x27;CHANNELS_LAST&#x27;,
    dilations=[],
    batch_dims=1,
    groups=1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

General function for computing a N-D convolution. It is required that
`1 <= N <= 3`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`.
Tensor of type T and shape `batch_shape + spatial_shape + [in_channels]` in the
case that `channels_last_format = true` or shape
`batch_shape + [in_channels] + spatial_shape` if `channels_last_format = false`.
spatial_shape is N-dimensional with `N=2` or `N=3`.
Also note that `batch_shape` is dictated by the parameter `batch_dims`
and defaults to 1.
</td>
</tr><tr>
<td>
`filter`<a id="filter"></a>
</td>
<td>
A `Tensor`. Must have the same type as `input`.
An `(N+2)-D` Tensor with the same type as `input` and shape
`spatial_filter_shape + [in_channels, out_channels]`, where spatial_filter_shape
is N-dimensional with `N=2` or `N=3`.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
A list of `ints`.
1-D tensor of length `N+2`. The stride of the sliding window for each
dimension of `input`. Must have `strides[0] = strides[N+1] = 1`.
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
An optional `string` from: `"CHANNELS_FIRST", "CHANNELS_LAST"`. Defaults to `"CHANNELS_LAST"`.
Used to set the data format. By default `CHANNELS_FIRST`, uses 
`NHWC (2D) / NDHWC (3D)` or if `CHANNELS_LAST`, uses `NCHW (2D) / NCDHW (3D)`.
</td>
</tr><tr>
<td>
`dilations`<a id="dilations"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
1-D tensor of length `N+2`. The dilation factor for each dimension of
`input`. If set to `k > 1`, there will be `k-1` skipped cells between each
filter element on that dimension. The dimension order is determined by the
value of `channels_last_format`, see above for details. Dilations in the batch
and depth dimensions must be 1.
</td>
</tr><tr>
<td>
`batch_dims`<a id="batch_dims"></a>
</td>
<td>
An optional `int`. Defaults to `1`.
A positive integer specifying the number of batch dimensions for the input
tensor. Should be less than the rank of the input tensor.
</td>
</tr><tr>
<td>
`groups`<a id="groups"></a>
</td>
<td>
An optional `int`. Defaults to `1`.
A positive integer specifying the number of groups in which the input is split
along the channel axis. Each group is convolved separately with
`filters / groups` filters. The output is the concatenation of all the groups
results along the channel axis. Input channels and filters must both be
divisible by groups.
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

