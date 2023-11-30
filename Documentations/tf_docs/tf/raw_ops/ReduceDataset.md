description: Reduces the input dataset to a singleton using a reduce function.
robots: noindex

# tf.raw_ops.ReduceDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Reduces the input dataset to a singleton using a reduce function.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ReduceDataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ReduceDataset(
    input_dataset,
    initial_state,
    other_arguments,
    f,
    output_types,
    output_shapes,
    use_inter_op_parallelism=True,
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
`input_dataset`<a id="input_dataset"></a>
</td>
<td>
A `Tensor` of type `variant`.
A variant tensor representing the input dataset.
</td>
</tr><tr>
<td>
`initial_state`<a id="initial_state"></a>
</td>
<td>
A list of `Tensor` objects.
A nested structure of tensors, representing the initial state of the
transformation.
</td>
</tr><tr>
<td>
`other_arguments`<a id="other_arguments"></a>
</td>
<td>
A list of `Tensor` objects.
</td>
</tr><tr>
<td>
`f`<a id="f"></a>
</td>
<td>
A function decorated with @Defun.
A function that maps `(old_state, input_element)` to `new_state`. It must take
two arguments and return a nested structures of tensors. The structure of
`new_state` must match the structure of `initial_state`.
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
`use_inter_op_parallelism`<a id="use_inter_op_parallelism"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
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
A list of `Tensor` objects of type `output_types`.
</td>
</tr>

</table>

