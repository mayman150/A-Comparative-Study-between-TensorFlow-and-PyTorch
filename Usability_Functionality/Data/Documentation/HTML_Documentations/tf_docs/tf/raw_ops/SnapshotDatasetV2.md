description: Creates a dataset that will write to / read from a snapshot.
robots: noindex

# tf.raw_ops.SnapshotDatasetV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a dataset that will write to / read from a snapshot.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.SnapshotDatasetV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.SnapshotDatasetV2(
    input_dataset,
    path,
    reader_func_other_args,
    shard_func_other_args,
    output_types,
    output_shapes,
    reader_func,
    shard_func,
    compression=&#x27;&#x27;,
    reader_prefix=&#x27;&#x27;,
    writer_prefix=&#x27;&#x27;,
    hash_valid=False,
    hash=0,
    metadata=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This dataset attempts to determine whether a valid snapshot exists at the
`snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
If not, it will run the preprocessing pipeline as usual, and write out a
snapshot of the data processed for future use.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_dataset`<a id="input_dataset"></a>
</td>
<td>
A `Tensor` of type `variant`.
A variant tensor representing the input dataset.
</td>
</tr><tr>
<td>
`path`<a id="path"></a>
</td>
<td>
A `Tensor` of type `string`.
The path we should write snapshots to / read snapshots from.
</td>
</tr><tr>
<td>
`reader_func_other_args`<a id="reader_func_other_args"></a>
</td>
<td>
A list of `Tensor` objects.
</td>
</tr><tr>
<td>
`shard_func_other_args`<a id="shard_func_other_args"></a>
</td>
<td>
A list of `Tensor` objects.
</td>
</tr><tr>
<td>
`output_types`<a id="output_types"></a>
</td>
<td>
A list of `tf.DTypes` that has length `>= 1`.
</td>
</tr><tr>
<td>
`output_shapes`<a id="output_shapes"></a>
</td>
<td>
A list of shapes (each a <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`) that has length `>= 1`.
</td>
</tr><tr>
<td>
`reader_func`<a id="reader_func"></a>
</td>
<td>
A function decorated with @Defun.
Optional. A function to control how to read data from snapshot shards.
</td>
</tr><tr>
<td>
`shard_func`<a id="shard_func"></a>
</td>
<td>
A function decorated with @Defun.
Optional. A function to control how to shard data when writing a snapshot.
</td>
</tr><tr>
<td>
`compression`<a id="compression"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
The type of compression to be applied to the saved snapshot files.
</td>
</tr><tr>
<td>
`reader_prefix`<a id="reader_prefix"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
</td>
</tr><tr>
<td>
`writer_prefix`<a id="writer_prefix"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
</td>
</tr><tr>
<td>
`hash_valid`<a id="hash_valid"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
</td>
</tr><tr>
<td>
`hash`<a id="hash"></a>
</td>
<td>
An optional `int`. Defaults to `0`.
</td>
</tr><tr>
<td>
`metadata`<a id="metadata"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
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
A `Tensor` of type `variant`.
</td>
</tr>

</table>

