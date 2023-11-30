description: Quantized Batch normalization.
robots: noindex

# tf.raw_ops.QuantizedBatchNormWithGlobalNormalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Quantized Batch normalization.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedBatchNormWithGlobalNormalization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedBatchNormWithGlobalNormalization(
    t,
    t_min,
    t_max,
    m,
    m_min,
    m_max,
    v,
    v_min,
    v_max,
    beta,
    beta_min,
    beta_max,
    gamma,
    gamma_min,
    gamma_max,
    out_type,
    variance_epsilon,
    scale_after_normalization,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This op is deprecated and will be removed in the future. Prefer
<a href="../../tf/nn/batch_normalization.md"><code>tf.nn.batch_normalization</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`t`<a id="t"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
A 4D input Tensor.
</td>
</tr><tr>
<td>
`t_min`<a id="t_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the lowest quantized input.
</td>
</tr><tr>
<td>
`t_max`<a id="t_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the highest quantized input.
</td>
</tr><tr>
<td>
`m`<a id="m"></a>
</td>
<td>
A `Tensor`. Must have the same type as `t`.
A 1D mean Tensor with size matching the last dimension of t.
This is the first output from tf.nn.moments,
or a saved moving average thereof.
</td>
</tr><tr>
<td>
`m_min`<a id="m_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the lowest quantized mean.
</td>
</tr><tr>
<td>
`m_max`<a id="m_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the highest quantized mean.
</td>
</tr><tr>
<td>
`v`<a id="v"></a>
</td>
<td>
A `Tensor`. Must have the same type as `t`.
A 1D variance Tensor with size matching the last dimension of t.
This is the second output from tf.nn.moments,
or a saved moving average thereof.
</td>
</tr><tr>
<td>
`v_min`<a id="v_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the lowest quantized variance.
</td>
</tr><tr>
<td>
`v_max`<a id="v_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the highest quantized variance.
</td>
</tr><tr>
<td>
`beta`<a id="beta"></a>
</td>
<td>
A `Tensor`. Must have the same type as `t`.
A 1D beta Tensor with size matching the last dimension of t.
An offset to be added to the normalized tensor.
</td>
</tr><tr>
<td>
`beta_min`<a id="beta_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the lowest quantized offset.
</td>
</tr><tr>
<td>
`beta_max`<a id="beta_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the highest quantized offset.
</td>
</tr><tr>
<td>
`gamma`<a id="gamma"></a>
</td>
<td>
A `Tensor`. Must have the same type as `t`.
A 1D gamma Tensor with size matching the last dimension of t.
If "scale_after_normalization" is true, this tensor will be multiplied
with the normalized tensor.
</td>
</tr><tr>
<td>
`gamma_min`<a id="gamma_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the lowest quantized gamma.
</td>
</tr><tr>
<td>
`gamma_max`<a id="gamma_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the highest quantized gamma.
</td>
</tr><tr>
<td>
`out_type`<a id="out_type"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
</td>
</tr><tr>
<td>
`variance_epsilon`<a id="variance_epsilon"></a>
</td>
<td>
A `float`. A small float number to avoid dividing by 0.
</td>
</tr><tr>
<td>
`scale_after_normalization`<a id="scale_after_normalization"></a>
</td>
<td>
A `bool`.
A bool indicating whether the resulted tensor
needs to be multiplied with gamma.
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
A tuple of `Tensor` objects (result, result_min, result_max).
</td>
</tr>
<tr>
<td>
`result`<a id="result"></a>
</td>
<td>
A `Tensor` of type `out_type`.
</td>
</tr><tr>
<td>
`result_min`<a id="result_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`result_max`<a id="result_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

