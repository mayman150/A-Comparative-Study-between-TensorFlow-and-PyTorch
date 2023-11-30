robots: noindex

# tf.raw_ops.DynamicEnqueueTPUEmbeddingRaggedTensorBatch

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
<p>`tf.compat.v1.raw_ops.DynamicEnqueueTPUEmbeddingRaggedTensorBatch`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DynamicEnqueueTPUEmbeddingRaggedTensorBatch(
    sample_splits,
    embedding_indices,
    aggregation_weights,
    mode_override,
    device_ordinal,
    table_ids,
    combiners=[],
    max_sequence_lengths=[],
    num_features=[],
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
`sample_splits`<a id="sample_splits"></a>
</td>
<td>
A list of at least 1 `Tensor` objects with the same type in: `int32`, `int64`.
</td>
</tr><tr>
<td>
`embedding_indices`<a id="embedding_indices"></a>
</td>
<td>
A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `int32`, `int64`.
</td>
</tr><tr>
<td>
`aggregation_weights`<a id="aggregation_weights"></a>
</td>
<td>
A list with the same length as `sample_splits` of `Tensor` objects with the same type in: `float32`, `float64`.
</td>
</tr><tr>
<td>
`mode_override`<a id="mode_override"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`device_ordinal`<a id="device_ordinal"></a>
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`table_ids`<a id="table_ids"></a>
</td>
<td>
A list of `ints`.
</td>
</tr><tr>
<td>
`combiners`<a id="combiners"></a>
</td>
<td>
An optional list of `strings`. Defaults to `[]`.
</td>
</tr><tr>
<td>
`max_sequence_lengths`<a id="max_sequence_lengths"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
</td>
</tr><tr>
<td>
`num_features`<a id="num_features"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`.
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
The created Operation.
</td>
</tr>

</table>

