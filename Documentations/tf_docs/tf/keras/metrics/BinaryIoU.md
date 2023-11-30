description: Computes the Intersection-Over-Union metric for class 0 and/or 1.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.BinaryIoU" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.BinaryIoU

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/iou_metrics.py#L310-L427">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the Intersection-Over-Union metric for class 0 and/or 1.

Inherits From: [`IoU`](../../../tf/keras/metrics/IoU.md), [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.BinaryIoU(
    target_class_ids: Union[List[int], Tuple[int, ...]] = (0, 1),
    threshold=0.5,
    name=None,
    dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

General definition and computation:

Intersection-Over-Union is a common evaluation metric for semantic image
segmentation.

For an individual class, the IoU metric is defined as follows:

```
iou = true_positives / (true_positives + false_positives + false_negatives)
```

To compute IoUs, the predictions are accumulated in a confusion matrix,
weighted by `sample_weight` and the metric is then calculated from it.

If `sample_weight` is `None`, weights default to 1.
Use `sample_weight` of 0 to mask values.

This class can be used to compute IoUs for a binary classification task
where the predictions are provided as logits. First a `threshold` is applied
to the predicted values such that those that are below the `threshold` are
converted to class 0 and those that are above the `threshold` are converted
to class 1.

IoUs for classes 0 and 1 are then computed, the mean of IoUs for the classes
that are specified by `target_class_ids` is returned.

Note: with `threshold=0`, this metric has the same behavior as `IoU`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target_class_ids`<a id="target_class_ids"></a>
</td>
<td>
A tuple or list of target class ids for which the metric
is returned. Options are `[0]`, `[1]`, or `[0, 1]`. With `[0]` (or
`[1]`), the IoU metric for class 0 (or class 1, respectively) is
returned. With `[0, 1]`, the mean of IoUs for the two classes is
returned.
</td>
</tr><tr>
<td>
`threshold`<a id="threshold"></a>
</td>
<td>
A threshold that applies to the prediction logits to convert
them to either predicted class 0 if the logit is below `threshold` or
predicted class 1 if the logit is above `threshold`.
</td>
</tr><tr>
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
</tr>
</table>



#### Standalone usage:



```
>>> m = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.3)
>>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7])
>>> m.result().numpy()
0.33333334
```

```
>>> m.reset_state()
>>> m.update_state([0, 1, 0, 1], [0.1, 0.2, 0.4, 0.7],
...                sample_weight=[0.2, 0.3, 0.4, 0.1])
>>> # cm = [[0.2, 0.4],
>>> #        [0.3, 0.1]]
>>> # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5],
>>> # true_positives = [0.2, 0.1]
>>> # iou = [0.222, 0.125]
>>> m.result().numpy()
0.17361112
```

Usage with `compile()` API:

```python
model.compile(
  optimizer='sgd',
  loss='mse',
  metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)])
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/iou_metrics.py#L150-L153">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/iou_metrics.py#L266-L295">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Compute the intersection-over-union via the confusion matrix.


<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/iou_metrics.py#L399-L419">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates the confusion matrix statistics.

Before the confusion matrix is updated, the predicted values are
thresholded to be:
  0 for values that are smaller than the `threshold`
  1 for values that are larger or equal to the `threshold`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
The ground truth values.
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional weighting of each example. Can
be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
and must be broadcastable to `y_true`. Defaults to `1`.
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





