description: Writes a generic summary to the default SummaryWriter if one exists.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.write" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.write

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Writes a generic summary to the default SummaryWriter if one exists.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.write(
    tag, tensor, step=None, metadata=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This exists primarily to support the definition of type-specific summary ops
like scalar() and image(), and is not intended for direct use unless defining
a new type-specific summary op.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tag`<a id="tag"></a>
</td>
<td>
string tag used to identify the summary (e.g. in TensorBoard), usually
generated with `tf.summary.summary_scope`
</td>
</tr><tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
the Tensor holding the summary data to write or a callable that
returns this Tensor. If a callable is passed, it will only be called when
a default SummaryWriter exists and the recording condition specified by
`record_if()` is met.
</td>
</tr><tr>
<td>
`step`<a id="step"></a>
</td>
<td>
Explicit `int64`-castable monotonic step value for this summary. If
omitted, this defaults to `tf.summary.experimental.get_step()`, which must
not be None.
</td>
</tr><tr>
<td>
`metadata`<a id="metadata"></a>
</td>
<td>
Optional SummaryMetadata, as a proto or serialized bytes
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional string name for this op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True on success, or false if no summary was written because no default
summary writer was available.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if a default writer exists, but no step was provided and
`tf.summary.experimental.get_step()` is None.
</td>
</tr>
</table>

