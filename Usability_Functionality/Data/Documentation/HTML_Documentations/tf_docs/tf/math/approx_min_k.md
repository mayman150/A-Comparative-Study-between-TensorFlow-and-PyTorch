description: Returns min k values and their indices of the input operand in an approximate manner.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.approx_min_k" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.approx_min_k

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Returns min `k` values and their indices of the input `operand` in an approximate manner.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.nn.approx_min_k`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.approx_min_k`, `tf.compat.v1.nn.approx_min_k`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.approx_min_k(
    operand,
    k,
    reduction_dimension=-1,
    recall_target=0.95,
    reduction_input_size_override=-1,
    aggregate_to_topk=True,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

See https://arxiv.org/abs/2206.14286 for the algorithm details. This op is
only optimized on TPU currently.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`operand`<a id="operand"></a>
</td>
<td>
Array to search for min-k. Must be a floating number type.
</td>
</tr><tr>
<td>
`k`<a id="k"></a>
</td>
<td>
Specifies the number of min-k.
</td>
</tr><tr>
<td>
`reduction_dimension`<a id="reduction_dimension"></a>
</td>
<td>
Integer dimension along which to search. Default: -1.
</td>
</tr><tr>
<td>
`recall_target`<a id="recall_target"></a>
</td>
<td>
Recall target for the approximation.
</td>
</tr><tr>
<td>
`reduction_input_size_override`<a id="reduction_input_size_override"></a>
</td>
<td>
When set to a positive value, it overrides
the size determined by `operand[reduction_dim]` for evaluating the recall.
This option is useful when the given `operand` is only a subset of the
overall computation in SPMD or distributed pipelines, where the true input
size cannot be deferred by the `operand` shape.
</td>
</tr><tr>
<td>
`aggregate_to_topk`<a id="aggregate_to_topk"></a>
</td>
<td>
When true, aggregates approximate results to top-k. When
false, returns the approximate results. The number of the approximate
results is implementation defined and is greater equals to the specified
`k`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Tuple of two arrays. The arrays are the least `k` values and the
corresponding indices along the `reduction_dimension` of the input
`operand`.  The arrays' dimensions are the same as the input `operand`
except for the `reduction_dimension`: when `aggregate_to_topk` is true,
the reduction dimension is `k`; otherwise, it is greater equals to `k`
where the size is implementation-defined.
</td>
</tr>

</table>


We encourage users to wrap `approx_min_k` with jit. See the following example
for nearest neighbor search over the squared l2 distance:

```
>>> import tensorflow as tf
>>> @tf.function(jit_compile=True)
... def l2_ann(qy, db, half_db_norms, k=10, recall_target=0.95):
...   dists = half_db_norms - tf.einsum('ik,jk->ij', qy, db)
...   return tf.nn.approx_min_k(dists, k=k, recall_target=recall_target)
>>>
>>> qy = tf.random.uniform((256,128))
>>> db = tf.random.uniform((2048,128))
>>> half_db_norms = tf.norm(db, axis=1) / 2
>>> dists, neighbors = l2_ann(qy, db, half_db_norms)
```

In the example above, we compute `db_norms/2 - dot(qy, db^T)` instead of
`qy^2 - 2 dot(qy, db^T) + db^2` for performance reason. The former uses less
arithmetics and produces the same set of neighbors.