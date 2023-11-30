description: Creates a Dataset that returns pseudorandom numbers.
robots: noindex

# tf.raw_ops.RandomDatasetV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a Dataset that returns pseudorandom numbers.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.RandomDatasetV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.RandomDatasetV2(
    seed,
    seed2,
    seed_generator,
    output_types,
    output_shapes,
    rerandomize_each_iteration=False,
    metadata=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Creates a Dataset that returns a stream of uniformly distributed
pseudorandom 64-bit signed integers. It accepts a boolean attribute that
determines if the random number generators are re-applied at each epoch. The
default value is True which means that the seeds are applied and the same
sequence of random numbers are generated at each epoch. If set to False, the
seeds are not re-applied and a different sequence of random numbers are
generated at each epoch.

In the TensorFlow Python API, you can instantiate this dataset via the
class `tf.data.experimental.RandomDatasetV2`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A `Tensor` of type `int64`.
A scalar seed for the random number generator. If either seed or
seed2 is set to be non-zero, the random number generator is seeded
by the given seed.  Otherwise, a random seed is used.
</td>
</tr><tr>
<td>
`seed2`<a id="seed2"></a>
</td>
<td>
A `Tensor` of type `int64`.
A second scalar seed to avoid seed collision.
</td>
</tr><tr>
<td>
`seed_generator`<a id="seed_generator"></a>
</td>
<td>
A `Tensor` of type `resource`.
A resource for the random number seed generator.
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
`rerandomize_each_iteration`<a id="rerandomize_each_iteration"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
A boolean attribute to rerandomize the sequence of random numbers generated
at each epoch.
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

