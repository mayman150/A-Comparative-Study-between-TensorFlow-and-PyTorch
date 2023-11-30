description: Creates a dataset that emits the records from one or more TFRecord files.
robots: noindex

# tf.raw_ops.TFRecordDatasetV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a dataset that emits the records from one or more TFRecord files.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TFRecordDatasetV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TFRecordDatasetV2(
    filenames,
    compression_type,
    buffer_size,
    byte_offsets,
    metadata=&#x27;&#x27;,
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
`filenames`<a id="filenames"></a>
</td>
<td>
A `Tensor` of type `string`.
A scalar or vector containing the name(s) of the file(s) to be
read.
</td>
</tr><tr>
<td>
`compression_type`<a id="compression_type"></a>
</td>
<td>
A `Tensor` of type `string`.
A scalar containing either (i) the empty string (no
compression), (ii) "ZLIB", or (iii) "GZIP".
</td>
</tr><tr>
<td>
`buffer_size`<a id="buffer_size"></a>
</td>
<td>
A `Tensor` of type `int64`.
A scalar representing the number of bytes to buffer. A value of
0 means no buffering will be performed.
</td>
</tr><tr>
<td>
`byte_offsets`<a id="byte_offsets"></a>
</td>
<td>
A `Tensor` of type `int64`.
A scalar or vector containing the number of bytes for each file
that will be skipped prior to reading.
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

