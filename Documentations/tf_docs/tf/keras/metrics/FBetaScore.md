description: Computes F-Beta score.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.FBetaScore" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.FBetaScore

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/f_score_metrics.py#L25-L247">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes F-Beta score.

Inherits From: [`Metric`](../../../tf/keras/metrics/Metric.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.FBetaScore(
    average=None,
    beta=1.0,
    threshold=None,
    name=&#x27;fbeta_score&#x27;,
    dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is the weighted harmonic mean of precision and recall.
Its output range is `[0, 1]`. It works for both multi-class
and multi-label classification.

#### It is defined as:



```python
b2 = beta ** 2
f_beta_score = (1 + b2) * (precision * recall) / (precision * b2 + recall)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`average`<a id="average"></a>
</td>
<td>
Type of averaging to be performed across per-class results
in the multi-class case.
Acceptable values are `None`, `"micro"`, `"macro"` and
`"weighted"`. Default value is `None`.
If `None`, no averaging is performed and `result()` will return
the score for each class.
If `"micro"`, compute metrics globally by counting the total
true positives, false negatives and false positives.
If `"macro"`, compute metrics for each label,
and return their unweighted mean.
This does not take label imbalance into account.
If `"weighted"`, compute metrics for each label,
and return their average weighted by support
(the number of true instances for each label).
This alters `"macro"` to account for label imbalance.
It can result in an score that is not between precision and recall.
</td>
</tr><tr>
<td>
`beta`<a id="beta"></a>
</td>
<td>
Determines the weight of given to recall
in the harmonic mean between precision and recall (see pseudocode
equation above). Default value is 1.
</td>
</tr><tr>
<td>
`threshold`<a id="threshold"></a>
</td>
<td>
Elements of `y_pred` greater than `threshold` are
converted to be 1, and the rest 0. If `threshold` is
`None`, the argmax of `y_pred` is converted to 1, and the rest to 0.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional. String name of the metric instance.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Optional. Data type of the metric result.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
F-Beta Score: float.
</td>
</tr>

</table>



#### Example:



```
>>> metric = tf.keras.metrics.FBetaScore(beta=2.0, threshold=0.5)
>>> y_true = np.array([[1, 1, 1],
...                    [1, 0, 0],
...                    [1, 1, 0]], np.int32)
>>> y_pred = np.array([[0.2, 0.6, 0.7],
...                    [0.2, 0.6, 0.6],
...                    [0.6, 0.8, 0.0]], np.float32)
>>> metric.update_state(y_true, y_pred)
>>> result = metric.result()
>>> result.numpy()
array([0.3846154 , 0.90909094, 0.8333334 ], dtype=float32)
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/f_score_metrics.py#L245-L247">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs/steps,
when a metric is evaluated during training.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/f_score_metrics.py#L208-L231">View source</a>

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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/f_score_metrics.py#L175-L206">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates statistics for the metric.

Note: This function is executed as a graph function in graph mode.
This means:
  a) Operations on the same resource are executed in textual order.
     This should make it easier to do things like add the updated
     value of a variable to another, for example.
  b) You don't need to worry about collecting the update ops to execute.
     All update ops added to the graph by this function will be
     executed.
  As a result, code should generally work the same way with graph or
  eager execution.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>

</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
A mini-batch of inputs to the Metric.
</td>
</tr>
</table>





