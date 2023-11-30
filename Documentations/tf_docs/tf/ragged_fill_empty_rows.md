<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.ragged_fill_empty_rows" />
<meta itemprop="path" content="Stable" />
</div>

# tf.ragged_fill_empty_rows

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
<p>`tf.compat.v1.ragged_fill_empty_rows`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.ragged_fill_empty_rows(
    value_rowids: Annotated[Any, _atypes.Int64],
    values: Annotated[Any, TV_RaggedFillEmptyRows_T],
    nrows: Annotated[Any, _atypes.Int64],
    default_value: Annotated[Any, TV_RaggedFillEmptyRows_T],
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
`value_rowids`<a id="value_rowids"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`values`<a id="values"></a>
</td>
<td>
A `Tensor`.
</td>
</tr><tr>
<td>
`nrows`<a id="nrows"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`default_value`<a id="default_value"></a>
</td>
<td>
A `Tensor`. Must have the same type as `values`.
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
A tuple of `Tensor` objects (output_value_rowids, output_values, empty_row_indicator, reverse_index_map).
</td>
</tr>
<tr>
<td>
`output_value_rowids`<a id="output_value_rowids"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`output_values`<a id="output_values"></a>
</td>
<td>
A `Tensor`. Has the same type as `values`.
</td>
</tr><tr>
<td>
`empty_row_indicator`<a id="empty_row_indicator"></a>
</td>
<td>
A `Tensor` of type `bool`.
</td>
</tr><tr>
<td>
`reverse_index_map`<a id="reverse_index_map"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr>
</table>

