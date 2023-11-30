description: Outputs whether the targets are in the top K predictions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.in_top_k" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.in_top_k

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Outputs whether the targets are in the top `K` predictions.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.nn.in_top_k`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.in_top_k(
    targets, predictions, k, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
prediction for the target class is finite (not inf, -inf, or nan) and among
the top `k` predictions among all predictions for example `i`.
`predictions` does not have to be normalized.

Note that the behavior of `InTopK` differs from the `TopK` op in its handling
of ties; if multiple classes have the same prediction value and straddle the
top-`k` boundary, all of those classes are considered to be in the top `k`.

```
>>> target = tf.constant([0, 1, 3])
>>> pred = tf.constant([
...  [1.2, -0.3, 2.8, 5.2],
...  [0.1, 0.0, 0.0, 0.0],
...  [0.0, 0.5, 0.3, 0.3]],
...  dtype=tf.float32)
>>> print(tf.math.in_top_k(target, pred, 2))
tf.Tensor([False  True  True], shape=(3,), dtype=bool)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`targets`<a id="targets"></a>
</td>
<td>
A `batch_size` vector of class ids. Must be `int32` or `int64`.
</td>
</tr><tr>
<td>
`predictions`<a id="predictions"></a>
</td>
<td>
A `batch_size` x `classes` tensor of type `float32`.
</td>
</tr><tr>
<td>
`k`<a id="k"></a>
</td>
<td>
An `int`. The parameter to specify search space.
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
A `Tensor` with the same shape of `targets` with type of `bool`. Each
element specifies if the target falls into top-k predictions.
</td>
</tr>

</table>

