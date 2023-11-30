description: Quantized Instance normalization.
robots: noindex

# tf.raw_ops.QuantizedInstanceNorm

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Quantized Instance normalization.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizedInstanceNorm`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizedInstanceNorm(
    x,
    x_min,
    x_max,
    output_range_given=False,
    given_y_min=0,
    given_y_max=0,
    variance_epsilon=1e-05,
    min_separation=0.001,
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
A 4D input Tensor.
</td>
</tr><tr>
<td>
`x_min`<a id="x_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the lowest quantized input.
</td>
</tr><tr>
<td>
`x_max`<a id="x_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
The value represented by the highest quantized input.
</td>
</tr><tr>
<td>
`output_range_given`<a id="output_range_given"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
If True, `given_y_min` and `given_y_min`
and `given_y_max` are used as the output range. Otherwise,
the implementation computes the output range.
</td>
</tr><tr>
<td>
`given_y_min`<a id="given_y_min"></a>
</td>
<td>
An optional `float`. Defaults to `0`.
Output in `y_min` if `output_range_given` is True.
</td>
</tr><tr>
<td>
`given_y_max`<a id="given_y_max"></a>
</td>
<td>
An optional `float`. Defaults to `0`.
Output in `y_max` if `output_range_given` is True.
</td>
</tr><tr>
<td>
`variance_epsilon`<a id="variance_epsilon"></a>
</td>
<td>
An optional `float`. Defaults to `1e-05`.
A small float number to avoid dividing by 0.
</td>
</tr><tr>
<td>
`min_separation`<a id="min_separation"></a>
</td>
<td>
An optional `float`. Defaults to `0.001`.
Minimum value of `y_max - y_min`
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
A tuple of `Tensor` objects (y, y_min, y_max).
</td>
</tr>
<tr>
<td>
`y`<a id="y"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`y_min`<a id="y_min"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`y_max`<a id="y_max"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

