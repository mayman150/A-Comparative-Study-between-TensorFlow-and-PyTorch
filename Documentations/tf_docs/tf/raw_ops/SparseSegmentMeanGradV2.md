description: Computes gradients for SparseSegmentMean.
robots: noindex

# tf.raw_ops.SparseSegmentMeanGradV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes gradients for SparseSegmentMean.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.SparseSegmentMeanGradV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.SparseSegmentMeanGradV2(
    grad, indices, segment_ids, dense_output_dim0, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is the number of unique indexes in "indices". Also returns vector
"sorted_unique_indices" containing the corresponding indexes from "indices".

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`grad`<a id="grad"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
gradient propagated to the SparseSegmentMean op.
</td>
</tr><tr>
<td>
`indices`<a id="indices"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
indices passed to the corresponding SparseSegmentMean op.
</td>
</tr><tr>
<td>
`segment_ids`<a id="segment_ids"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
segment_ids passed to the corresponding SparseSegmentMean op.
</td>
</tr><tr>
<td>
`dense_output_dim0`<a id="dense_output_dim0"></a>
</td>
<td>
A `Tensor` of type `int32`.
dimension 0 of "data" passed to SparseSegmentMean op.
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
A tuple of `Tensor` objects (output, sorted_unique_indices).
</td>
</tr>
<tr>
<td>
`output`<a id="output"></a>
</td>
<td>
A `Tensor`. Has the same type as `grad`.
</td>
</tr><tr>
<td>
`sorted_unique_indices`<a id="sorted_unique_indices"></a>
</td>
<td>
A `Tensor`. Has the same type as `indices`.
</td>
</tr>
</table>

