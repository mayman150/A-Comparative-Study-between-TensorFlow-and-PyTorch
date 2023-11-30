description: Computes QuantizedConv2D per channel.
robots: noindex

# tf.raw_ops.QuantizedConv2DPerChannel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes QuantizedConv2D per channel.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedConv2DPerChannel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedConv2DPerChannel(
    input,
    filter,
    min_input,
    max_input,
    min_filter,
    max_filter,
    strides,
    padding,
    out_type=<a href="../../tf/dtypes.md#qint32"><code>tf.dtypes.qint32</code></a>,
    dilations=[1, 1, 1, 1],
    name=None
)
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
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
The original input tensor.
</td>
</tr><tr>
<td>
`filter`<a id="filter"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
The original filter tensor.
</td>
</tr><tr>
<td>
`min_input`<a id="min_input"></a>
</td>
<td>
A `Tensor` of type `float32`.
The minimum value of the input tensor
</td>
</tr><tr>
<td>
`max_input`<a id="max_input"></a>
</td>
<td>
A `Tensor` of type `float32`.
The maximum value of the input tensor.
</td>
</tr><tr>
<td>
`min_filter`<a id="min_filter"></a>
</td>
<td>
A `Tensor` of type `float32`.
The minimum value of the filter tensor.
</td>
</tr><tr>
<td>
`max_filter`<a id="max_filter"></a>
</td>
<td>
A `Tensor` of type `float32`.
The maximum value of the filter tensor.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
A list of `ints`. list of stride values.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
A `string` from: `"SAME", "VALID"`.
</td>
</tr><tr>
<td>
`out_type`<a id="out_type"></a>
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to <a href="../../tf.md#qint32"><code>tf.qint32</code></a>.
The quantized type of output tensor that needs to be converted.
</td>
</tr><tr>
<td>
`dilations`<a id="dilations"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
list of dilation values.
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
A tuple of `Tensor` objects (output, min_output, max_output).
</td>
</tr>
<tr>
<td>
`output`<a id="output"></a>
</td>
<td>
A `Tensor` of type `out_type`.
</td>
</tr><tr>
<td>
`min_output`<a id="min_output"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_output`<a id="max_output"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

