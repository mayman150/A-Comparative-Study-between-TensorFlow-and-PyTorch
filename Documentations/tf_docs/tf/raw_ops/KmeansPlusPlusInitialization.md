description: Selects num_to_sample rows of input using the KMeans++ criterion.
robots: noindex

# tf.raw_ops.KmeansPlusPlusInitialization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Selects num_to_sample rows of input using the KMeans++ criterion.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.KmeansPlusPlusInitialization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.KmeansPlusPlusInitialization(
    points, num_to_sample, seed, num_retries_per_sample, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Rows of points are assumed to be input points. One row is selected at random.
Subsequent rows are sampled with probability proportional to the squared L2
distance from the nearest row selected thus far till num_to_sample rows have
been sampled.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`points`<a id="points"></a>
</td>
<td>
A `Tensor` of type `float32`.
Matrix of shape (n, d). Rows are assumed to be input points.
</td>
</tr><tr>
<td>
`num_to_sample`<a id="num_to_sample"></a>
</td>
<td>
A `Tensor` of type `int64`.
Scalar. The number of rows to sample. This value must not be larger than n.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A `Tensor` of type `int64`.
Scalar. Seed for initializing the random number generator.
</td>
</tr><tr>
<td>
`num_retries_per_sample`<a id="num_retries_per_sample"></a>
</td>
<td>
A `Tensor` of type `int64`.
Scalar. For each row that is sampled, this parameter
specifies the number of additional points to draw from the current
distribution before selecting the best. If a negative value is specified, a
heuristic is used to sample O(log(num_to_sample)) additional points.
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
A `Tensor` of type `float32`.
</td>
</tr>

</table>

