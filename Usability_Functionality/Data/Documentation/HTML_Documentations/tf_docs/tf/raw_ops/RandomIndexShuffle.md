description: Outputs the position of value in a permutation of [0, ..., max_index].
robots: noindex

# tf.raw_ops.RandomIndexShuffle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Outputs the position of `value` in a permutation of [0, ..., max_index].


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.RandomIndexShuffle`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.RandomIndexShuffle(
    index, seed, max_index, rounds=4, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Output values are a bijection of the `index` for any combination and `seed` and `max_index`.

If multiple inputs are vectors (matrix in case of seed) then the size of the
first dimension must match.

The outputs are deterministic.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`index`<a id="index"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `uint32`, `int64`, `uint64`.
A scalar tensor or a vector of dtype `dtype`. The index (or indices) to be shuffled. Must be within [0, max_index].
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `uint32`, `int64`, `uint64`.
A tensor of dtype `Tseed` and shape [3] or [n, 3]. The random seed.
</td>
</tr><tr>
<td>
`max_index`<a id="max_index"></a>
</td>
<td>
A `Tensor`. Must have the same type as `index`.
A scalar tensor or vector of dtype `dtype`. The upper bound(s) of the interval (inclusive).
</td>
</tr><tr>
<td>
`rounds`<a id="rounds"></a>
</td>
<td>
An optional `int`. Defaults to `4`.
The number of rounds to use the in block cipher.
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
A `Tensor`. Has the same type as `index`.
</td>
</tr>

</table>

