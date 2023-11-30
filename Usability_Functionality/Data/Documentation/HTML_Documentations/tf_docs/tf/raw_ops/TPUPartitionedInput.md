description: An op that groups a list of partitioned inputs together.
robots: noindex

# tf.raw_ops.TPUPartitionedInput

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



An op that groups a list of partitioned inputs together.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TPUPartitionedInput`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TPUPartitionedInput(
    inputs, partition_dim=0, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->
 This op

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A list of at least 1 `Tensor` objects with the same type.
A list of partitioned inputs which must have the same shape.
</td>
</tr><tr>
<td>
`partition_dim`<a id="partition_dim"></a>
</td>
<td>
An optional `int`. Defaults to `0`.
An integer describles which dimension is partitioned. -1 means
those inputs are replicated.
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
A `Tensor`. Has the same type as `inputs`.
</td>
</tr>

</table>

