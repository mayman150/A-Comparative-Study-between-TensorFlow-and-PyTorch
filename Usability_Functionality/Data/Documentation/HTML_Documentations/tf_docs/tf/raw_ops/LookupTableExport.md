description: Outputs all keys and values in the table.
robots: noindex

# tf.raw_ops.LookupTableExport

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Outputs all keys and values in the table.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.LookupTableExport`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.LookupTableExport(
    table_handle, Tkeys, Tvalues, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`table_handle`<a id="table_handle"></a>
</td>
<td>
A `Tensor` of type mutable `string`. Handle to the table.
</td>
</tr><tr>
<td>
`Tkeys`<a id="Tkeys"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>.
</td>
</tr><tr>
<td>
`Tvalues`<a id="Tvalues"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>.
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
A tuple of `Tensor` objects (keys, values).
</td>
</tr>
<tr>
<td>
`keys`<a id="keys"></a>
</td>
<td>
A `Tensor` of type `Tkeys`.
</td>
</tr><tr>
<td>
`values`<a id="values"></a>
</td>
<td>
A `Tensor` of type `Tvalues`.
</td>
</tr>
</table>

