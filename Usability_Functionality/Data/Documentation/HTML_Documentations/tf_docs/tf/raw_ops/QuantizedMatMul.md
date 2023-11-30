description: Perform a quantized matrix multiplication of  a by the matrix b.
robots: noindex

# tf.raw_ops.QuantizedMatMul

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Perform a quantized matrix multiplication of  `a` by the matrix `b`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedMatMul`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedMatMul(
    a,
    b,
    min_a,
    max_a,
    min_b,
    max_b,
    Toutput=<a href="../../tf/dtypes.md#qint32"><code>tf.dtypes.qint32</code></a>,
    transpose_a=False,
    transpose_b=False,
    Tactivation=<a href="../../tf/dtypes.md#quint8"><code>tf.dtypes.quint8</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The inputs must be two-dimensional matrices and the inner dimension of
`a` (after being transposed if `transpose_a` is non-zero) must match the
outer dimension of `b` (after being transposed if `transposed_b` is
non-zero).

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
Must be a two-dimensional tensor.
</td>
</tr><tr>
<td>
`b`<a id="b"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
Must be a two-dimensional tensor.
</td>
</tr><tr>
<td>
`min_a`<a id="min_a"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the lowest quantized `a` value represents.
</td>
</tr><tr>
<td>
`max_a`<a id="max_a"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the highest quantized `a` value represents.
</td>
</tr><tr>
<td>
`min_b`<a id="min_b"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the lowest quantized `b` value represents.
</td>
</tr><tr>
<td>
`max_b`<a id="max_b"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the highest quantized `b` value represents.
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
`transpose_a`<a id="transpose_a"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
If true, `a` is transposed before multiplication.
</td>
</tr><tr>
<td>
`transpose_b`<a id="transpose_b"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
If true, `b` is transposed before multiplication.
</td>
</tr><tr>
<td>
`Tactivation`<a id="Tactivation"></a>
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to <a href="../../tf.md#quint8"><code>tf.quint8</code></a>.
The type of output produced by activation function
following this operation.
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

