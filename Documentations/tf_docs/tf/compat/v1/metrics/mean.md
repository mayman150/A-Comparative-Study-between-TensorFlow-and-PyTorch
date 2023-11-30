description: Computes the (weighted) mean of the given values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.metrics.mean" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.metrics.mean

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/metrics_impl.py">View source</a>



Computes the (weighted) mean of the given values.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.metrics.mean(
    values,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../../tf/compat/v1/metrics/mean.md"><code>tf.compat.v1.metrics.mean</code></a> is not compatible with eager
execution or <a href="../../../../tf/function.md"><code>tf.function</code></a>.
Please use <a href="../../../../tf/keras/metrics/Mean.md"><code>tf.keras.metrics.Mean</code></a> instead for TF2 migration. After
instantiating a <a href="../../../../tf/keras/metrics/Mean.md"><code>tf.keras.metrics.Mean</code></a> object, you can first call the
`update_state()` method to record the new values, and then call the
`result()` method to get the mean eagerly. You can also attach it to a
Keras model with the `add_metric` method.  Please refer to the [migration
guide](https://www.tensorflow.org/guide/migrate#new-style_metrics_and_losses)
for more details.

#### Structural Mapping to TF2

Before:

```python
mean, update_op = tf.compat.v1.metrics.mean(
  values=values,
  weights=weights,
  metrics_collections=metrics_collections,
  update_collections=update_collections,
  name=name)
```

After:

```python
 m = tf.keras.metrics.Mean(
   name=name)

 m.update_state(
   values=values,
   sample_weight=weights)

 mean = m.result()
```

#### How to Map Arguments

| TF1 Arg Name          | TF2 Arg Name    | Note                       |
| :-------------------- | :-------------- | :------------------------- |
| `values`              | `values`        | In `update_state()` method |
| `weights`             | `sample_weight` | In `update_state()` method |
| `metrics_collections` | Not supported   | Metrics should be tracked  |
:                       :                 : explicitly or with Keras   :
:                       :                 : APIs, for example,         :
:                       :                 : [add_metric][add_metric],  :
:                       :                 : instead of via collections :
| `updates_collections` | Not supported   | -                          |
| `name`                | `name`          | In constructor             |

[add_metric]:https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_metric


#### Before & After Usage Example

Before:

```
>>> g = tf.Graph()
>>> with g.as_default():
...   values = [1, 2, 3]
...   mean, update_op = tf.compat.v1.metrics.mean(values)
...   global_init = tf.compat.v1.global_variables_initializer()
...   local_init = tf.compat.v1.local_variables_initializer()
>>> sess = tf.compat.v1.Session(graph=g)
>>> sess.run([global_init, local_init])
>>> sess.run(update_op)
>>> sess.run(mean)
2.0
```


After:

```
>>> m = tf.keras.metrics.Mean()
>>> m.update_state([1, 2, 3])
>>> m.result().numpy()
2.0
```

```python
# Used within Keras model
model.add_metric(tf.keras.metrics.Mean()(values))
```


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

The `mean` function creates two local variables, `total` and `count`
that are used to compute the average of `values`. This average is ultimately
returned as `mean` which is an idempotent operation that simply divides
`total` by `count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `mean`.
`update_op` increments `total` with the reduced sum of the product of `values`
and `weights`, and it increments `count` with the reduced sum of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`values`<a id="values"></a>
</td>
<td>
A `Tensor` of arbitrary dimensions.
</td>
</tr><tr>
<td>
`weights`<a id="weights"></a>
</td>
<td>
Optional `Tensor` whose rank is either 0, or the same rank as
`values`, and must be broadcastable to `values` (i.e., all dimensions must
be either `1`, or the same as the corresponding `values` dimension).
</td>
</tr><tr>
<td>
`metrics_collections`<a id="metrics_collections"></a>
</td>
<td>
An optional list of collections that `mean`
should be added to.
</td>
</tr><tr>
<td>
`updates_collections`<a id="updates_collections"></a>
</td>
<td>
An optional list of collections that `update_op`
should be added to.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
An optional variable_scope name.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`mean`<a id="mean"></a>
</td>
<td>
A `Tensor` representing the current mean, the value of `total` divided
by `count`.
</td>
</tr><tr>
<td>
`update_op`<a id="update_op"></a>
</td>
<td>
An operation that increments the `total` and `count` variables
appropriately and whose value matches `mean_value`.
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
If `weights` is not `None` and its shape doesn't match `values`,
or if either `metrics_collections` or `updates_collections` are not a list
or tuple.
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


