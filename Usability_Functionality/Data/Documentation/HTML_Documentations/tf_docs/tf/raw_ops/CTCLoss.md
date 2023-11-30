description: Calculates the CTC Loss (log probability) for each batch entry.
robots: noindex

# tf.raw_ops.CTCLoss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Calculates the CTC Loss (log probability) for each batch entry.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.CTCLoss`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.CTCLoss(
    inputs,
    labels_indices,
    labels_values,
    sequence_length,
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=False,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->
  Also calculates

the gradient.  This class performs the softmax operation for you, so inputs
should be e.g. linear projections of outputs by an LSTM.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`.
3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
</td>
</tr><tr>
<td>
`labels_indices`<a id="labels_indices"></a>
</td>
<td>
A `Tensor` of type `int64`.
The indices of a `SparseTensor<int32, 2>`.
`labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
`(batch b, time t)`.
</td>
</tr><tr>
<td>
`labels_values`<a id="labels_values"></a>
</td>
<td>
A `Tensor` of type `int32`.
The values (labels) associated with the given batch and time.
</td>
</tr><tr>
<td>
`sequence_length`<a id="sequence_length"></a>
</td>
<td>
A `Tensor` of type `int32`.
A vector containing sequence lengths (batch).
</td>
</tr><tr>
<td>
`preprocess_collapse_repeated`<a id="preprocess_collapse_repeated"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
Scalar, if true then repeated labels are
collapsed prior to the CTC calculation.
</td>
</tr><tr>
<td>
`ctc_merge_repeated`<a id="ctc_merge_repeated"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
Scalar.  If set to false, *during* CTC calculation
repeated non-blank labels will not be merged and are interpreted as
individual labels.  This is a simplified version of CTC.
</td>
</tr><tr>
<td>
`ignore_longer_outputs_than_inputs`<a id="ignore_longer_outputs_than_inputs"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
Scalar. If set to true, during CTC
calculation, items that have longer output sequences than input sequences
are skipped: they don't contribute to the loss term and have zero-gradient.
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
A tuple of `Tensor` objects (loss, gradient).
</td>
</tr>
<tr>
<td>
`loss`<a id="loss"></a>
</td>
<td>
A `Tensor`. Has the same type as `inputs`.
</td>
</tr><tr>
<td>
`gradient`<a id="gradient"></a>
</td>
<td>
A `Tensor`. Has the same type as `inputs`.
</td>
</tr>
</table>

