description: An op that groups a list of partitioned inputs together. Supports ND sharding.
robots: noindex

# tf.raw_ops.TPUPartitionedInputV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



An op that groups a list of partitioned inputs together. Supports ND sharding.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TPUPartitionedInputV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TPUPartitionedInputV2(
    inputs, partition_dims, is_packed=False, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


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
`partition_dims`<a id="partition_dims"></a>
</td>
<td>
A list of `ints`.
A list of integers describing how each dimension is partitioned. Emptiness
indicates the inputs are replicated.
</td>
</tr><tr>
<td>
`is_packed`<a id="is_packed"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
Indicates whether the input is a packed resource.
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

