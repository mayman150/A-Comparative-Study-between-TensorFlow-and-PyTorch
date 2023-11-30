robots: noindex

# tf.raw_ops.QuantizedMatMulWithBiasAndRequantize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>






<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedMatMulWithBiasAndRequantize`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedMatMulWithBiasAndRequantize(
    a,
    b,
    bias,
    min_a,
    max_a,
    min_b,
    max_b,
    min_freezed_output,
    max_freezed_output,
    Toutput=<a href="../../tf/dtypes.md#quint8"><code>tf.dtypes.quint8</code></a>,
    transpose_a=False,
    transpose_b=False,
    input_quant_mode=&#x27;MIN_FIRST&#x27;,
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
`a`<a id="a"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`b`<a id="b"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`bias`<a id="bias"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `qint32`.
</td>
</tr><tr>
<td>
`min_a`<a id="min_a"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_a`<a id="max_a"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`min_b`<a id="min_b"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_b`<a id="max_b"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`min_freezed_output`<a id="min_freezed_output"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_freezed_output`<a id="max_freezed_output"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`Toutput`<a id="Toutput"></a>
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to <a href="../../tf.md#quint8"><code>tf.quint8</code></a>.
</td>
</tr><tr>
<td>
`transpose_a`<a id="transpose_a"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
</td>
</tr><tr>
<td>
`transpose_b`<a id="transpose_b"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
</td>
</tr><tr>
<td>
`input_quant_mode`<a id="input_quant_mode"></a>
</td>
<td>
An optional `string` from: `"MIN_FIRST", "SCALED"`. Defaults to `"MIN_FIRST"`.
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
A tuple of `Tensor` objects (out, min_out, max_out).
</td>
</tr>
<tr>
<td>
`out`<a id="out"></a>
</td>
<td>
A `Tensor` of type `Toutput`.
</td>
</tr><tr>
<td>
`min_out`<a id="min_out"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_out`<a id="max_out"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

