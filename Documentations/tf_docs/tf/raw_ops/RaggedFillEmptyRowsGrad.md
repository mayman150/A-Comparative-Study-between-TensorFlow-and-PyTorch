robots: noindex

# tf.raw_ops.RaggedFillEmptyRowsGrad

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
<p>`tf.compat.v1.raw_ops.RaggedFillEmptyRowsGrad`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.RaggedFillEmptyRowsGrad(
    reverse_index_map, grad_values, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`reverse_index_map`<a id="reverse_index_map"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`grad_values`<a id="grad_values"></a>
</td>
<td>
A `Tensor`.
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
A tuple of `Tensor` objects (d_values, d_default_value).
</td>
</tr>
<tr>
<td>
`d_values`<a id="d_values"></a>
</td>
<td>
A `Tensor`. Has the same type as `grad_values`.
</td>
</tr><tr>
<td>
`d_default_value`<a id="d_default_value"></a>
</td>
<td>
A `Tensor`. Has the same type as `grad_values`.
</td>
</tr>
</table>

