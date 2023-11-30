description: A preprocessing layer which buckets continuous features by ranges.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Discretization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="compile"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="update_state"/>
</div>

# tf.keras.layers.Discretization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/discretization.py#L136-L444">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which buckets continuous features by ranges.

Inherits From: [`PreprocessingLayer`](../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Discretization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Discretization(
    bin_boundaries=None,
    num_bins=None,
    epsilon=0.01,
    output_mode=&#x27;int&#x27;,
    sparse=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will place each element of its input data into one of several
contiguous ranges and output an integer index indicating which range each
element was placed in.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Any <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> or <a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> of dimension 2 or higher.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Same as input shape.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`bin_boundaries`<a id="bin_boundaries"></a>
</td>
<td>
A list of bin boundaries. The leftmost and rightmost bins
will always extend to `-inf` and `inf`, so `bin_boundaries=[0., 1., 2.]`
generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +inf)`.
If this option is set, `adapt()` should not be called.
</td>
</tr><tr>
<td>
`num_bins`<a id="num_bins"></a>
</td>
<td>
The integer number of bins to compute. If this option is set,
`adapt()` should be called to learn the bin boundaries.
</td>
</tr><tr>
<td>
`epsilon`<a id="epsilon"></a>
</td>
<td>
Error tolerance, typically a small fraction close to zero (e.g.
0.01). Higher values of epsilon increase the quantile approximation, and
hence result in more unequal buckets, but could improve performance
and resource consumption.
</td>
</tr><tr>
<td>
`output_mode`<a id="output_mode"></a>
</td>
<td>
Specification for the output of the layer. Values can be
`"int"`, `"one_hot"`, `"multi_hot"`, or
 `"count"` configuring the layer as follows:
   - `"int"`: Return the discretized bin indices directly.
   - `"one_hot"`: Encodes each individual element in the input into an
     array the same size as `num_bins`, containing a 1 at the input's bin
     index. If the last dimension is size 1, will encode on that
     dimension.  If the last dimension is not size 1, will append a new
     dimension for the encoded output.
   - `"multi_hot"`: Encodes each sample in the input into a single array
     the same size as `num_bins`, containing a 1 for each bin index
     index present in the sample. Treats the last dimension as the sample
     dimension, if input shape is `(..., sample_length)`, output shape
     will be `(..., num_tokens)`.
   - `"count"`: As `"multi_hot"`, but the int array contains a count of
     the number of times the bin index appeared in the sample.
 Defaults to `"int"`.
</td>
</tr><tr>
<td>
`sparse`<a id="sparse"></a>
</td>
<td>
Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
and `"count"` output modes. If True, returns a `SparseTensor` instead of
a dense `Tensor`. Defaults to `False`.
</td>
</tr>
</table>



#### Examples:



Bucketize float values based on provided buckets.
```
>>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
>>> layer = tf.keras.layers.Discretization(bin_boundaries=[0., 1., 2.])
>>> layer(input)
<tf.Tensor: shape=(2, 4), dtype=int64, numpy=
array([[0, 2, 3, 1],
       [1, 3, 2, 1]])>
```

Bucketize float values based on a number of buckets to compute.
```
>>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
>>> layer = tf.keras.layers.Discretization(num_bins=4, epsilon=0.01)
>>> layer.adapt(input)
>>> layer(input)
<tf.Tensor: shape=(2, 4), dtype=int64, numpy=
array([[0, 2, 3, 2],
       [1, 3, 3, 1]])>
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/discretization.py#L315-L361">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adapt(
    data, batch_size=None, steps=None
)
</code></pre>

Computes bin boundaries from quantiles in a input dataset.

Calling `adapt()` on a `Discretization` layer is an alternative to
passing in a `bin_boundaries` argument during construction. A
`Discretization` layer should always be either adapted over a dataset or
passed `bin_boundaries`.

During `adapt()`, the layer will estimate the quantile boundaries of the
input dataset. The number of quantiles can be controlled via the
`num_bins` argument, and the error tolerance for quantile boundaries can
be controlled via the `epsilon` argument.

In order to make `Discretization` efficient in any distribution context,
the computed boundaries are kept static with respect to any compiled
<a href="../../../tf/Graph.md"><code>tf.Graph</code></a>s that call the layer. As a consequence, if the layer is
adapted a second time, any models using the layer should be re-compiled.
For more information see
<a href="../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md#adapt"><code>tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt</code></a>.

`adapt()` is meant only as a single machine utility to compute layer
state.  To analyze a dataset that cannot fit on a single machine, see
[Tensorflow Transform](
https://www.tensorflow.org/tfx/transform/get_started) for a
multi-machine, map-reduce solution.

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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/discretization.py#L391-L395">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets the statistics of the preprocessing layer.


<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/discretization.py#L363-L380">View source</a>

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





