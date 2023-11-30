description: Returns the index of a data point that should be added to the seed set.
robots: noindex

# tf.raw_ops.KMC2ChainInitialization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns the index of a data point that should be added to the seed set.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.KMC2ChainInitialization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.KMC2ChainInitialization(
    distances, seed, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Entries in distances are assumed to be squared distances of candidate points to
the already sampled centers in the seed set. The op constructs one Markov chain
of the k-MC^2 algorithm and returns the index of one candidate point to be added
as an additional cluster center.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`distances`<a id="distances"></a>
</td>
<td>
A `Tensor` of type `float32`.
Vector with squared distances to the closest previously sampled cluster center
for each candidate point.
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
A `Tensor` of type `int64`.
</td>
</tr>

</table>

