description: Writes a tensor summary.
robots: noindex

# tf.raw_ops.WriteSummary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Writes a tensor summary.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.WriteSummary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.WriteSummary(
    writer, step, tensor, tag, summary_metadata, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Writes `tensor` at `step` with `tag` using summary `writer`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`writer`<a id="writer"></a>
</td>
<td>
A `Tensor` of type `resource`.
</td>
</tr><tr>
<td>
`step`<a id="step"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
A `Tensor`.
</td>
</tr><tr>
<td>
`tag`<a id="tag"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`summary_metadata`<a id="summary_metadata"></a>
</td>
<td>
A `Tensor` of type `string`.
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

