description: Computes the Intersection-Over-Union metric for one-hot encoded labels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.OneHotIoU" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.OneHotIoU

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/iou_metrics.py#L535-L648">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the Intersection-Over-Union metric for one-hot encoded labels.

Inherits From: [`IoU`](../../../tf/keras/metrics/IoU.md), [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.OneHotIoU(
    num_classes: int,
    target_class_ids: Union[List[int], Tuple[int, ...]],
    name=None,
    dtype=None,
    ignore_class: Optional[int] = None,
    sparse_y_pred: bool = False,
    axis: int = -1
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

This class can be used to compute IoU for multi-class classification tasks
where the labels are one-hot encoded (the last axis should have one
dimension per class). Note that the predictions should also have the same
shape. To compute the IoU, first the labels and predictions are converted
back into integer format by taking the argmax over the class axis. Then the
same computation steps as for the base `IoU` class apply.

Note, if there is only one channel in the labels and predictions, this class
is the same as class `IoU`. In this case, use `IoU` instead.

Also, make sure that `num_classes` is equal to the number of classes in the
data, to avoid a "labels out of bound" error when the confusion matrix is
computed.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_classes`<a id="num_classes"></a>
</td>
<td>
The possible number of labels the prediction task can have.
A confusion matrix of shape `(num_classes, num_classes)` will be
allocated to accumulate predictions from which the metric is calculated.
</td>
</tr><tr>
<td>
`target_class_ids`<a id="target_class_ids"></a>
</td>
<td>
A tuple or list of target class ids for which the metric
is returned. To compute IoU for a specific class, a list (or tuple) of a
single id value should be provided.
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
</tr><tr>
<td>
`ignore_class`<a id="ignore_class"></a>
</td>
<td>
Optional integer. The ID of a class to be ignored during
metric computation. This is useful, for example, in segmentation
problems featuring a "void" class (commonly -1 or 255) in segmentation
maps. By default (`ignore_class=None`), all classes are considered.
</td>
</tr><tr>
<td>
`sparse_y_pred`<a id="sparse_y_pred"></a>
</td>
<td>
Whether predictions are encoded using natural numbers or
probability distribution vectors. If `False`, the <a href="../../../tf/math/argmax.md"><code>tf.argmax</code></a> function
will be used to determine each sample's most likely associated label.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
(Optional) The dimension containing the logits. Defaults to `-1`.
</td>
</tr>
</table>



#### Standalone usage:



```
>>> y_true = tf.constant([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
>>> y_pred = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.5, 0.3, 0.1],
...                       [0.1, 0.4, 0.5]])
>>> sample_weight = [0.1, 0.2, 0.3, 0.4]
>>> m = tf.keras.metrics.OneHotIoU(num_classes=3, target_class_ids=[0, 2])
>>> m.update_state(
...     y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
>>> # cm = [[0, 0, 0.2+0.4],
>>> #       [0.3, 0, 0],
>>> #       [0, 0, 0.1]]
>>> # sum_row = [0.3, 0, 0.7], sum_col = [0.6, 0.3, 0.1]
>>> # true_positives = [0, 0, 0.1]
>>> # single_iou = true_positives / (sum_row + sum_col - true_positives))
>>> # mean_iou = (0 / (0.3 + 0.6 - 0) + 0.1 / (0.7 + 0.1 - 0.1)) / 2
>>> m.result().numpy()
0.071
```

Usage with `compile()` API:

```python
model.compile(
  optimizer='sgd',
  loss='mse',
  metrics=[tf.keras.metrics.OneHotIoU(num_classes=3, target_class_id=[1])])
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/iou_metrics.py#L98-L148">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates the confusion matrix statistics.


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





