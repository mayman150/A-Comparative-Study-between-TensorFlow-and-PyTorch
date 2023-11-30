description: Lookup embedding results, accounting for invalid IDs and empty features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.safe_embedding_lookup_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.safe_embedding_lookup_sparse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/embedding_ops.py">View source</a>



Lookup embedding results, accounting for invalid IDs and empty features.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner=&#x27;mean&#x27;,
    default_id=None,
    max_norm=None,
    name=None,
    allow_fast_lookup=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

The partitioned embedding in `embedding_weights` must all be the same shape
except for the first dimension. The first dimension is allowed to vary as the
vocabulary size is not necessarily a multiple of num of shards.

This is similar to <a href="../../tf/nn/embedding_lookup_sparse.md"><code>tf.nn.embedding_lookup_sparse</code></a>, except invalid IDs (< 0)
are pruned from input IDs and weights, as well as any IDs with non-positive
weight. For an entry with no features, the embedding vector for `default_id`
is returned, or the 0-vector if `default_id` is not supplied. See
<a href="../../tf/nn/embedding_lookup_sparse.md"><code>tf.nn.embedding_lookup_sparse</code></a> for more information on how sparse embedding
lookups work in general.

The ids and weights may be multi-dimensional `SparseTensor`s or
`RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
yield higher performance.

If `len(embedding_weights) > 1`, each element `id` of `ids` is partitioned
between the elements of `embedding_weights` according to the "div" partition
strategy, which means we assign ids to partitions in a contiguous manner. For
instance, 13 ids are split across 5 partitions as:
`[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

If the id space does not evenly divide the number of partitions, each of the
first `(max_id + 1) % len(embedding_weights)` partitions will be assigned one
more id.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`embedding_weights`<a id="embedding_weights"></a>
</td>
<td>
A single tensor representing the complete embedding
tensor, or a list of tensors all of same shape except for the first
dimension, representing sharded embedding tensors following "div"
partition strategy.
</td>
</tr><tr>
<td>
`sparse_ids`<a id="sparse_ids"></a>
</td>
<td>
`SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
ids, where `d_0` is typically batch size, or a `RaggedTensor` with rank 2.
</td>
</tr><tr>
<td>
`sparse_weights`<a id="sparse_weights"></a>
</td>
<td>
`SparseTensor` or `RaggedTensor` of same type and shape as
`sparse_ids`, containing float weights corresponding to `sparse_ids`, or
`None` if all weights are assumed to be 1.0.
</td>
</tr><tr>
<td>
`combiner`<a id="combiner"></a>
</td>
<td>
A string specifying how to combine embedding results for each
entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
default.
</td>
</tr><tr>
<td>
`default_id`<a id="default_id"></a>
</td>
<td>
The id to use for an entry with no features. Defaults to
0-vector.
</td>
</tr><tr>
<td>
`max_norm`<a id="max_norm"></a>
</td>
<td>
If not `None`, all embeddings are l2-normalized to max_norm before
combining.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this operation (optional).
</td>
</tr><tr>
<td>
`allow_fast_lookup`<a id="allow_fast_lookup"></a>
</td>
<td>
An optional boolean specifying whether to allow
simplified embedding lookups when `params` is a single tensor and
`max_norm` is `None`. Setting this flag to `True` during training can
cause the use of dense gradients with increased memory footprint.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dense tensor representing the combined embeddings for the
sparse ids. For each row in the dense tensor represented by `sparse_ids`,
the op looks up the embeddings for all ids in that row, multiplies them by
the corresponding weight, and combines these embeddings as specified.

In other words, if

  `shape(combined embedding_weights) = [p0, p1, ..., pm]`

and

  `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

then

  `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

  ```python
  [0, 0]: id 1, weight 2.0
  [0, 1]: id 3, weight 0.5
  [1, 0]: id -1, weight 1.0
  [2, 3]: id 1, weight 3.0
  ```

`default_id` is 0.

with `combiner`="mean", then the output will be a 3x20 matrix where

  ```python
  output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
  output[1, :] = (params[0, :] * 1.0) / 1.0
  output[2, :] = (params[1, :] * 3.0) / 3.0
  ```
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
if `embedding_weights` is empty.
</td>
</tr>
</table>

