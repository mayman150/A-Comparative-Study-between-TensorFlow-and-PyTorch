description: A preprocessing layer which normalizes continuous features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Normalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="compile"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.layers.Normalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/normalization.py#L29-L393">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which normalizes continuous features.

Inherits From: [`PreprocessingLayer`](../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Normalization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Normalization(
    axis=-1, mean=None, variance=None, invert=False, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will shift and scale inputs into a distribution centered around
0 with standard deviation 1. It accomplishes this by precomputing the mean
and variance of the data, and calling `(input - mean) / sqrt(var)` at
runtime.

The mean and variance values for the layer must be either supplied on
construction or learned via `adapt()`. `adapt()` will compute the mean and
variance of the data and store them as the layer's weights. `adapt()` should
be called before `fit()`, `evaluate()`, or `predict()`.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
Integer, tuple of integers, or None. The axis or axes that should
have a separate mean and variance for each index in the shape. For
example, if shape is `(None, 5)` and `axis=1`, the layer will track 5
separate mean and variance values for the last axis. If `axis` is set
to `None`, the layer will normalize all elements in the input by a
scalar mean and variance. When `-1` the last axis of the
input is assumed to be a feature dimension and is normalized per
index. Note that in the specific case of batched scalar inputs where
the only axis is the batch axis, the default will normalize each index
in the batch separately. In this case, consider passing `axis=None`.
Defaults to `-1`.
</td>
</tr><tr>
<td>
`mean`<a id="mean"></a>
</td>
<td>
The mean value(s) to use during normalization. The passed value(s)
will be broadcast to the shape of the kept axes above; if the value(s)
cannot be broadcast, an error will be raised when this layer's
`build()` method is called.
</td>
</tr><tr>
<td>
`variance`<a id="variance"></a>
</td>
<td>
The variance value(s) to use during normalization. The passed
value(s) will be broadcast to the shape of the kept axes above; if the
value(s) cannot be broadcast, an error will be raised when this
layer's `build()` method is called.
</td>
</tr><tr>
<td>
`invert`<a id="invert"></a>
</td>
<td>
If True, this layer will apply the inverse transformation
to its inputs: it would turn a normalized input back into its
original form.
</td>
</tr>
</table>



#### Examples:



Calculate a global mean and variance by analyzing the dataset in `adapt()`.

```
>>> adapt_data = np.array([1., 2., 3., 4., 5.], dtype='float32')
>>> input_data = np.array([1., 2., 3.], dtype='float32')
>>> layer = tf.keras.layers.Normalization(axis=None)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
<tf.Tensor: shape=(3,), dtype=float32, numpy=
array([-1.4142135, -0.70710677, 0.], dtype=float32)>
```

Calculate a mean and variance for each index on the last axis.

```
>>> adapt_data = np.array([[0., 7., 4.],
...                        [2., 9., 6.],
...                        [0., 7., 4.],
...                        [2., 9., 6.]], dtype='float32')
>>> input_data = np.array([[0., 7., 4.]], dtype='float32')
>>> layer = tf.keras.layers.Normalization(axis=-1)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=
array([-1., -1., -1.], dtype=float32)>
```

Pass the mean and variance directly.

```
>>> input_data = np.array([[1.], [2.], [3.]], dtype='float32')
>>> layer = tf.keras.layers.Normalization(mean=3., variance=2.)
>>> layer(input_data)
<tf.Tensor: shape=(3, 1), dtype=float32, numpy=
array([[-1.4142135 ],
       [-0.70710677],
       [ 0.        ]], dtype=float32)>
```

Use the layer to de-normalize inputs (after adapting the layer).

```
>>> adapt_data = np.array([[0., 7., 4.],
...                        [2., 9., 6.],
...                        [0., 7., 4.],
...                        [2., 9., 6.]], dtype='float32')
>>> input_data = np.array([[1., 2., 3.]], dtype='float32')
>>> layer = tf.keras.layers.Normalization(axis=-1, invert=True)
>>> layer.adapt(adapt_data)
>>> layer(input_data)
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=
array([2., 10., 8.], dtype=float32)>
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`is_adapted`<a id="is_adapted"></a>
</td>
<td>
Whether the layer has been fit to data already.
</td>
</tr>
</table>



## Methods

<h3 id="adapt"><code>adapt</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/normalization.py#L241-L287">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adapt(
    data, batch_size=None, steps=None
)
</code></pre>

Computes the mean and variance of values in a dataset.

Calling `adapt()` on a `Normalization` layer is an alternative to
passing in `mean` and `variance` arguments during layer construction. A
`Normalization` layer should always either be adapted over a dataset or
passed `mean` and `variance`.

During `adapt()`, the layer will compute a `mean` and `variance`
separately for each position in each axis specified by the `axis`
argument. To calculate a single `mean` and `variance` over the input
data, simply pass `axis=None`.

In order to make `Normalization` efficient in any distribution context,
the computed mean and variance are kept static with respect to any
compiled <a href="../../../tf/Graph.md"><code>tf.Graph</code></a>s that call the layer. As a consequence, if the layer
is adapted a second time, any models using the layer should be
re-compiled. For more information see
<a href="../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md#adapt"><code>tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt</code></a>.

`adapt()` is meant only as a single machine utility to compute layer
state.  To analyze a dataset that cannot fit on a single machine, see
[Tensorflow Transform](
https://www.tensorflow.org/tfx/transform/get_started)
for a multi-machine, map-reduce solution.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
The data to train on. It can be passed either as a
<a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>, or as a numpy array.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`.
Number of samples per state update.
If unspecified, `batch_size` will default to 32.
Do not specify the `batch_size` if your data is in the
form of datasets, generators, or <a href="../../../tf/keras/utils/Sequence.md"><code>keras.utils.Sequence</code></a> instances
(since they generate batches).
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Integer or `None`.
Total number of steps (batches of samples)
When training with input tensors such as
TensorFlow data tensors, the default `None` is equal to
the number of samples in your dataset divided by
the batch size, or 1 if that cannot be determined. If x is a
<a href="../../../tf/data.md"><code>tf.data</code></a> dataset, and 'steps' is None, the epoch will run until
the input dataset is exhausted. When passing an infinitely
repeating dataset, you must specify the `steps` argument. This
argument is not supported with array inputs.
</td>
</tr>
</table>



<h3 id="compile"><code>compile</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/base_preprocessing_layer.py#L139-L160">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile(
    run_eagerly=None, steps_per_execution=None
)
</code></pre>

Configures the layer for `adapt`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`run_eagerly`
</td>
<td>
Bool. If `True`, this `Model`'s
logic will not be wrapped in a <a href="../../../tf/function.md"><code>tf.function</code></a>. Recommended to leave
this as `None` unless your `Model` cannot be run inside a
<a href="../../../tf/function.md"><code>tf.function</code></a>. Defaults to `False`.
</td>
</tr><tr>
<td>
`steps_per_execution`
</td>
<td>
Int. The number of batches to run
during each <a href="../../../tf/function.md"><code>tf.function</code></a> call. Running multiple batches inside a
single <a href="../../../tf/function.md"><code>tf.function</code></a> call can greatly improve performance on TPUs or
small models with a large Python overhead. Defaults to `1`.
</td>
</tr>
</table>



<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/normalization.py#L333-L339">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets the statistics of the preprocessing layer.


<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/normalization.py#L289-L331">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    data
)
</code></pre>

Accumulates statistics for the preprocessing layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
A mini-batch of inputs to the layer.
</td>
</tr>
</table>





