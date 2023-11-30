robots: noindex

# tf.raw_ops.ChooseFastestBranchDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>






<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ChooseFastestBranchDataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ChooseFastestBranchDataset(
    input_dataset,
    ratio_numerator,
    ratio_denominator,
    other_arguments,
    num_elements_per_branch,
    branches,
    other_arguments_lengths,
    output_types,
    output_shapes,
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
</td>
</tr><tr>
<td>
`ratio_numerator`<a id="ratio_numerator"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`ratio_denominator`<a id="ratio_denominator"></a>
</td>
<td>
A `Tensor` of type `int64`.
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
`num_elements_per_branch`<a id="num_elements_per_branch"></a>
</td>
<td>
An `int` that is `>= 1`.
</td>
</tr><tr>
<td>
`branches`<a id="branches"></a>
</td>
<td>
A list of functions decorated with @Defun that has length `>= 1`.
</td>
</tr><tr>
<td>
`other_arguments_lengths`<a id="other_arguments_lengths"></a>
</td>
<td>
A list of `ints` that has length `>= 1`.
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

