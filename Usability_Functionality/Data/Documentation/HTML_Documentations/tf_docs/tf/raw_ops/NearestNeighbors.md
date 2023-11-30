description: Selects the k nearest centers for each point.
robots: noindex

# tf.raw_ops.NearestNeighbors

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Selects the k nearest centers for each point.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.NearestNeighbors`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.NearestNeighbors(
    points, centers, k, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Rows of points are assumed to be input points. Rows of centers are assumed to be
the list of candidate centers. For each point, the k centers that have least L2
distance to it are computed.

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
`centers`<a id="centers"></a>
</td>
<td>
A `Tensor` of type `float32`.
Matrix of shape (m, d). Rows are assumed to be centers.
</td>
</tr><tr>
<td>
`k`<a id="k"></a>
</td>
<td>
A `Tensor` of type `int64`.
Number of nearest centers to return for each point. If k is larger than m, then
only m centers are returned.
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
A tuple of `Tensor` objects (nearest_center_indices, nearest_center_distances).
</td>
</tr>
<tr>
<td>
`nearest_center_indices`<a id="nearest_center_indices"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`nearest_center_distances`<a id="nearest_center_distances"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

