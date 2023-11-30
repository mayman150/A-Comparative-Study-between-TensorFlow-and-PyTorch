description: Computes the crossentropy metric between the labels and predictions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.CategoricalCrossentropy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.CategoricalCrossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/probabilistic_metrics.py#L168-L237">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the crossentropy metric between the labels and predictions.

Inherits From: [`MeanMetricWrapper`](../../../tf/keras/metrics/MeanMetricWrapper.md), [`Mean`](../../../tf/keras/metrics/Mean.md), [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.CategoricalCrossentropy(
    name=&#x27;categorical_crossentropy&#x27;,
    dtype=None,
    from_logits=False,
    label_smoothing=0,
    axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the crossentropy metric class to be used when there are multiple
label classes (2 or more). Here we assume that labels are given as a
`one_hot` representation. eg., When labels values are [2, 0, 1],
 `y_true` = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
(Optional) string name of the metric instance.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
(Optional) data type of the metric result.
</td>
</tr><tr>
<td>
`from_logits`<a id="from_logits"></a>
</td>
<td>
(Optional) Whether output is expected to be a logits tensor.
By default, we consider that output encodes a probability distribution.
</td>
</tr><tr>
<td>
`label_smoothing`<a id="label_smoothing"></a>
</td>
<td>
(Optional) Float in [0, 1]. When > 0, label values are
smoothed, meaning the confidence on label values are relaxed. e.g.
`label_smoothing=0.2` means that we will use a value of `0.1` for label
`0` and `0.9` for label `1`"
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
(Optional) -1 is the dimension along which entropy is
computed. Defaults to `-1`.
</td>
</tr>
</table>



#### Standalone usage:



```
>>> # EPSILON = 1e-7, y = y_true, y` = y_pred
>>> # y` = clip_ops.clip_by_value(output, EPSILON, 1. - EPSILON)
>>> # y` = [[0.05, 0.95, EPSILON], [0.1, 0.8, 0.1]]
>>> # xent = -sum(y * log(y'), axis = -1)
>>> #      = -((log 0.95), (log 0.1))
>>> #      = [0.051, 2.302]
>>> # Reduced xent = (0.051 + 2.302) / 2
>>> m = tf.keras.metrics.CategoricalCrossentropy()
>>> m.update_state([[0, 1, 0], [0, 0, 1]],
...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
>>> m.result().numpy()
1.1769392
```

```
>>> m.reset_state()
>>> m.update_state([[0, 1, 0], [0, 0, 1]],
...                [[0.05, 0.95, 0], [0.1, 0.8, 0.1]],
...                sample_weight=tf.constant([0.3, 0.7]))
>>> m.result().numpy()
1.6271976
```

Usage with `compile()` API:

```python
model.compile(
  optimizer='sgd',
  loss='categorical_crossentropy',
  metrics=[tf.keras.metrics.CategoricalCrossentropy()])
```

## Methods

<h3 id="merge_state"><code>merge_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/base_metric.py#L288-L326">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_state(
    metrics
)
</code></pre>

Merges the state from one or more metrics.

This method can be used by distributed systems to merge the state
computed by different metric instances. Typically the state will be
stored in the form of the metric's weights. For example, a
tf.keras.metrics.Mean metric contains a list of two weight values: a
total and a count. If there were two instances of a
tf.keras.metrics.Accuracy that each independently aggregated partial
state for an overall accuracy calculation, these two metric's states
could be combined as follows:

```
>>> m1 = tf.keras.metrics.Accuracy()
>>> _ = m1.update_state([[1], [2]], [[0], [2]])
```

```
>>> m2 = tf.keras.metrics.Accuracy()
>>> _ = m2.update_state([[3], [4]], [[3], [4]])
```

```
>>> m2.merge_state([m1])
>>> m2.result().numpy()
0.75
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`metrics`
</td>
<td>
an iterable of metrics. The metrics must have compatible
state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided iterable does not contain metrics matching
the metric's required specifications.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/base_metric.py#L249-L265">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/base_metric.py#L551-L563">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Computes and returns the scalar metric value tensor or a dict of scalars.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar tensor, or a dictionary of scalar tensors.
</td>
</tr>

</table>



<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/base_metric.py#L686-L728">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates metric statistics.

For sparse categorical metrics, the shapes of `y_true` and `y_pred` are
different.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
Ground truth label values. shape = `[batch_size, d0, .. dN-1]` or
shape = `[batch_size, d0, .. dN-1, 1]`.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted probability values. shape = `[batch_size, d0, .. dN]`.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional `sample_weight` acts as a
coefficient for the metric. If a scalar is provided, then the metric is
simply scaled by the given value. If `sample_weight` is a tensor of size
`[batch_size]`, then the metric for each sample of the batch is rescaled
by the corresponding element in the `sample_weight` vector. If the shape
of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
to this shape), then each metric element of `y_pred` is scaled by the
corresponding value of `sample_weight`. (Note on `dN-1`: all metric
functions reduce by 1 dimension, usually the last axis (-1)).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Update op.
</td>
</tr>

</table>





