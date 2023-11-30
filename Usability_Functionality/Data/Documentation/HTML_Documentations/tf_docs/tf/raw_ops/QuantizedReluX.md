description: Computes Quantized Rectified Linear X: min(max(features, 0), max_value)
robots: noindex

# tf.raw_ops.QuantizedReluX

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedReluX`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedReluX(
    features,
    max_value,
    min_features,
    max_features,
    out_type=<a href="../../tf/dtypes.md#quint8"><code>tf.dtypes.quint8</code></a>,
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
`features`<a id="features"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
</td>
</tr><tr>
<td>
`max_value`<a id="max_value"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`min_features`<a id="min_features"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the lowest quantized value represents.
</td>
</tr><tr>
<td>
`max_features`<a id="max_features"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the highest quantized value represents.
</td>
</tr><tr>
<td>
`out_type`<a id="out_type"></a>
</td>
<td>
An optional <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to <a href="../../tf.md#quint8"><code>tf.quint8</code></a>.
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
A tuple of `Tensor` objects (activations, min_activations, max_activations).
</td>
</tr>
<tr>
<td>
`activations`<a id="activations"></a>
</td>
<td>
A `Tensor` of type `out_type`.
</td>
</tr><tr>
<td>
`min_activations`<a id="min_activations"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_activations`<a id="max_activations"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

