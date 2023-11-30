description: Metric which runs in Python, compiled outside of the TensorFlow graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.experimental.PyMetric" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="merge_state"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.metrics.experimental.PyMetric

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/py_metric.py#L25-L191">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Metric which runs in Python, compiled outside of the TensorFlow graph.

Inherits From: [`Metric`](../../../../tf/keras/metrics/Metric.md), [`Layer`](../../../../tf/keras/layers/Layer.md), [`Module`](../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.experimental.PyMetric(
    name=None, dtype=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
(Optional) string name of the PyMetric instance.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
(Optional) data type of the PyMetric result.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Additional layer keywords arguments.
</td>
</tr>
</table>


Usage of `PyMetric` is generally identical to <a href="../../../../tf/keras/metrics/Metric.md"><code>keras.metrics.Metric</code></a>.
It can be used in isolation, or in tandem with the `compile()` API. For more
information about the usage of `PyMetric`, see <a href="../../../../tf/keras/metrics/Metric.md"><code>keras.metrics.Metric</code></a>.

Unlike regular metrics, `PyMetric` instances are outside-compiled
with respect to the TensorFlow graph during training or evaluation.
They have access to the same
inputs of a standard in-graph metric, but they run in a Python interpreter
on the host CPU. Any data stored in a `PyMetric` is located on the main
memory of the host CPU, and any TensorFlow ops used in a PyMetric are
run eagerly on the host CPU.

As a result, `PyMetric` instances are generally not as performant
as in-graph metrics, and should only be used in cases where computing
the metric inside of the TensorFlow graph is either impossible
or prohibitively expensive.

**Note:** Due to the use of <a href="../../../../tf/py_function.md"><code>tf.py_function</code></a>, PyMetrics
are incompatible with XLA and therefore TPUs.

Methods to be implemented by subclasses:

* `update_state()`: Handles updates to internal state variables
* `result()`: Computes and returns a scalar value or a dict of scalar values
  for the metric from the state variables.
* `reset_state()`: Computes and returns a scalar value for the metric from
  the state variables.

This subclass implementation is similar to that of <a href="../../../../tf/keras/metrics/Metric.md"><code>keras.metrics.Metric</code></a>,
with two notable differences:

* Inputs to `update_state()` in a `PyMetric` are eager tensors, and both
`update_state()` and `result()` run outside of the TensorFlow graph,
executing any TensorFlow ops eagerly.
* `reset_state()` is also called at initialization time to initialize the
Python state of the metric.
* `result()` can only return a single scalar. It does not support returning
a dictionary of results like <a href="../../../../tf/keras/metrics/Metric.md"><code>keras.metrics.Metric</code></a>.

Example subclass implementation using sklearn's Jaccard Score:

```python
from sklearn.metrics import jaccard_score
import tensorflow as tf

class JaccardScore(tf.keras.metrics.experimental.PyMetric):

  def __init__(self, name='jaccard_score', **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.jaccard_sum += jaccard_score(y_pred, y_true, average="macro")
    self.count += 1

  def reset_state(self):
    self.jaccard_sum = 0.
    self.count = 0.

  def result(self):
    return self.jaccard_sum / self.count
```

## Methods

<h3 id="merge_state"><code>merge_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/py_metric.py#L161-L168">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merge_state(
    metrics
)
</code></pre>

Merges the state from one or more metrics.

`PyMetric` instances that intend to support merging state must override
 this method, as the default implementation
in <a href="../../../../tf/keras/metrics/Metric.md"><code>keras.metrics.Metric</code></a> does not apply to `PyMetric`.

<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/py_metric.py#L170-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets all of the metric state variables.

This function is called between epochs when a metric is evaluated during
training. It's also called when the metric is initialized.

<h3 id="result"><code>result</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/py_metric.py#L178-L191">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>result()
</code></pre>

Computes and returns the scalar metric value.

**Note:** This function is executed outside of the TensorFlow graph
 on the CPU host. This means any TensorFlow ops run in this method
 are run eagerly.

Result computation is an idempotent operation that simply calculates the
metric value using the state variables.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Python scalar.
</td>
</tr>

</table>



<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/metrics/py_metric.py#L141-L159">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Accumulates statistics for the metric.

**Note:** This function is executed outside of the TensorFlow graph
on the CPU host.

#### This means:



a) Inputs are eager tensors.
b) Any TensorFlow ops run in this method are run eagerly.
c) Any Tensors created are allocated to the CPU's main memory.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
Target output
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
Predicted output
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
(Optional) weights for the individual samples in
`y_true` and `y_pred`
</td>
</tr>
</table>





