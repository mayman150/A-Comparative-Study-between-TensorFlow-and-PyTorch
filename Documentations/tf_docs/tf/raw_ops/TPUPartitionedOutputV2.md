description: An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned
robots: noindex

# tf.raw_ops.TPUPartitionedOutputV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



An op that demultiplexes a tensor to be sharded by XLA to a list of partitioned


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TPUPartitionedOutputV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TPUPartitionedOutputV2(
    inputs, num_splits, partition_dims, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


outputs outside the XLA computation. Supports ND sharding.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A `Tensor`.
A tensor which represents the full shape of partitioned tensors.
</td>
</tr><tr>
<td>
`num_splits`<a id="num_splits"></a>
</td>
<td>
An `int` that is `>= 1`.
</td>
</tr><tr>
<td>
`partition_dims`<a id="partition_dims"></a>
</td>
<td>
A list of `ints`.
A list of integers describing how each dimension is partitioned. Emptiness
indicates the inputs are replicated.
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
A list of `num_splits` `Tensor` objects with the same type as `inputs`.
</td>
</tr>

</table>

