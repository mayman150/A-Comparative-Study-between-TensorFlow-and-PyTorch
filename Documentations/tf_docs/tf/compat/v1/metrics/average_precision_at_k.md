description: Computes average precision@k of predictions with respect to sparse labels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.metrics.average_precision_at_k" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.metrics.average_precision_at_k

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/metrics_impl.py">View source</a>



Computes average precision@k of predictions with respect to sparse labels.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.metrics.average_precision_at_k(
    labels,
    predictions,
    k,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`average_precision_at_k` creates two local variables,
`average_precision_at_<k>/total` and `average_precision_at_<k>/max`, that
are used to compute the frequency. This frequency is ultimately returned as
`average_precision_at_<k>`: an idempotent operation that simply divides
`average_precision_at_<k>/total` by `average_precision_at_<k>/max`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
indicating the top `k` `predictions`. Set operations applied to `top_k` and
`labels` calculate the true positives and false positives weighted by
`weights`. Then `update_op` increments `true_positive_at_<k>` and
`false_positive_at_<k>` using these values.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
`int64` `Tensor` or `SparseTensor` with shape
[D1, ... DN, num_labels] or [D1, ... DN], where the latter implies
num_labels=1. N >= 1 and num_labels is the number of target classes for
the associated prediction. Commonly, N=1 and `labels` has shape
[batch_size, num_labels]. [D1, ... DN] must match `predictions`. Values
should be in range [0, num_classes), where num_classes is the last
dimension of `predictions`. Values outside this range are ignored.
</td>
</tr><tr>
<td>
`predictions`<a id="predictions"></a>
</td>
<td>
Float `Tensor` with shape [D1, ... DN, num_classes] where
N >= 1. Commonly, N=1 and `predictions` has shape
[batch size, num_classes]. The final dimension contains the logit values
for each class. [D1, ... DN] must match `labels`.
</td>
</tr><tr>
<td>
`k`<a id="k"></a>
</td>
<td>
Integer, k for @k metric. This will calculate an average precision for
range `[1,k]`, as documented above.
</td>
</tr><tr>
<td>
`weights`<a id="weights"></a>
</td>
<td>
`Tensor` whose rank is either 0, or n-1, where n is the rank of
`labels`. If the latter, it must be broadcastable to `labels` (i.e., all
dimensions must be either `1`, or the same as the corresponding `labels`
dimension).
</td>
</tr><tr>
<td>
`metrics_collections`<a id="metrics_collections"></a>
</td>
<td>
An optional list of collections that values should
be added to.
</td>
</tr><tr>
<td>
`updates_collections`<a id="updates_collections"></a>
</td>
<td>
An optional list of collections that updates should
be added to.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Name of new update operation, and namespace for other dependent ops.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`mean_average_precision`<a id="mean_average_precision"></a>
</td>
<td>
Scalar `float64` `Tensor` with the mean average
precision values.
</td>
</tr><tr>
<td>
`update`<a id="update"></a>
</td>
<td>
`Operation` that increments variables appropriately, and whose
value matches `metric`.
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
if k is invalid.
</td>
</tr><tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
If eager execution is enabled.
</td>
</tr>
</table>

