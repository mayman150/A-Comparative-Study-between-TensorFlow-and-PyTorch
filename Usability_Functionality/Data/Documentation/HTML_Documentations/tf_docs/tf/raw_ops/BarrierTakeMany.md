description: Takes the given number of completed elements from a barrier.
robots: noindex

# tf.raw_ops.BarrierTakeMany

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Takes the given number of completed elements from a barrier.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.BarrierTakeMany`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.BarrierTakeMany(
    handle,
    num_elements,
    component_types,
    allow_small_batch=False,
    wait_for_incomplete=False,
    timeout_ms=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation concatenates completed-element component tensors along
the 0th dimension to make a single component tensor.

Elements come out of the barrier when they are complete, and in the order
in which they were placed into the barrier.  The indices output provides
information about the batch in which each element was originally inserted
into the barrier.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`handle`<a id="handle"></a>
</td>
<td>
A `Tensor` of type mutable `string`. The handle to a barrier.
</td>
</tr><tr>
<td>
`num_elements`<a id="num_elements"></a>
</td>
<td>
A `Tensor` of type `int32`.
A single-element tensor containing the number of elements to
take.
</td>
</tr><tr>
<td>
`component_types`<a id="component_types"></a>
</td>
<td>
A list of `tf.DTypes` that has length `>= 1`.
The type of each component in a value.
</td>
</tr><tr>
<td>
`allow_small_batch`<a id="allow_small_batch"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
Allow to return less than num_elements items if barrier is
already closed.
</td>
</tr><tr>
<td>
`wait_for_incomplete`<a id="wait_for_incomplete"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
</td>
</tr><tr>
<td>
`timeout_ms`<a id="timeout_ms"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
If the queue is empty, this operation will block for up to
timeout_ms milliseconds.
Note: This option is not supported yet.
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
A tuple of `Tensor` objects (indices, keys, values).
</td>
</tr>
<tr>
<td>
`indices`<a id="indices"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`keys`<a id="keys"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`values`<a id="values"></a>
</td>
<td>
A list of `Tensor` objects of type `component_types`.
</td>
</tr>
</table>

