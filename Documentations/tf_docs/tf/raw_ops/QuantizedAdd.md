description: Returns x + y element-wise, working on quantized buffers.
robots: noindex

# tf.raw_ops.QuantizedAdd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns x + y element-wise, working on quantized buffers.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedAdd`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedAdd(
    x,
    y,
    min_x,
    max_x,
    min_y,
    max_y,
    Toutput=<a href="../../tf/dtypes.md#qint32"><code>tf.dtypes.qint32</code></a>,
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
`x`<a id="x"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`y`<a id="y"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`min_x`<a id="min_x"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the lowest quantized `x` value represents.
</td>
</tr><tr>
<td>
`max_x`<a id="max_x"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the highest quantized `x` value represents.
</td>
</tr><tr>
<td>
`min_y`<a id="min_y"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the lowest quantized `y` value represents.
</td>
</tr><tr>
<td>
`max_y`<a id="max_y"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the highest quantized `y` value represents.
</td>
</tr><tr>
<td>
`Toutput`<a id="Toutput"></a>
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to <a href="../../tf.md#qint32"><code>tf.qint32</code></a>.
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
A tuple of `Tensor` objects (z, min_z, max_z).
</td>
</tr>
<tr>
<td>
`z`<a id="z"></a>
</td>
<td>
A `Tensor` of type `Toutput`.
</td>
</tr><tr>
<td>
`min_z`<a id="min_z"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_z`<a id="max_z"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

