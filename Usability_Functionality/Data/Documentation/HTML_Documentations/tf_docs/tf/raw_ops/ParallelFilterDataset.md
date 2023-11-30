description: Creates a dataset containing elements of input_dataset matching predicate.
robots: noindex

# tf.raw_ops.ParallelFilterDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a dataset containing elements of `input_dataset` matching `predicate`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ParallelFilterDataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ParallelFilterDataset(
    input_dataset,
    other_arguments,
    num_parallel_calls,
    predicate,
    output_types,
    output_shapes,
    deterministic=&#x27;default&#x27;,
    metadata=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The `predicate` function must return a scalar boolean and accept the
following arguments:

* One tensor for each component of an element of `input_dataset`.
* One tensor for each value in `other_arguments`.

Unlike a "FilterDataset", which applies `predicate` sequentially, this dataset
invokes up to `num_parallel_calls` copies of `predicate` in parallel.

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
</td>
</tr><tr>
<td>
`other_arguments`<a id="other_arguments"></a>
</td>
<td>
A list of `Tensor` objects.
A list of tensors, typically values that were captured when
building a closure for `predicate`.
</td>
</tr><tr>
<td>
`num_parallel_calls`<a id="num_parallel_calls"></a>
</td>
<td>
A `Tensor` of type `int64`.
The number of concurrent invocations of `predicate` that process
elements from `input_dataset` in parallel.
</td>
</tr><tr>
<td>
`predicate`<a id="predicate"></a>
</td>
<td>
A function decorated with @Defun.
A function returning a scalar boolean.
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
`deterministic`<a id="deterministic"></a>
</td>
<td>
An optional `string`. Defaults to `"default"`.
A string indicating the op-level determinism to use. Deterministic controls
whether the interleave is allowed to return elements out of order if the next
element to be returned isn't available, but a later element is. Options are
"true", "false", and "default". "default" indicates that determinism should be
decided by the `experimental_deterministic` parameter of <a href="../../tf/data/Options.md"><code>tf.data.Options</code></a>.
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

