description: Layer to be used as an entry point into a Network (a graph of layers).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.InputLayer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.InputLayer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/input_layer.py#L43-L292">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Layer to be used as an entry point into a Network (a graph of layers).

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.InputLayer(
    input_shape=None,
    batch_size=None,
    dtype=None,
    input_tensor=None,
    sparse=None,
    name=None,
    ragged=None,
    type_spec=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

It can either wrap an existing tensor (pass an `input_tensor` argument)
or create a placeholder tensor (pass arguments `input_shape`, and
optionally, `dtype`).

It is generally recommend to use the Keras Functional model via `Input`,
(which creates an `InputLayer`) without directly using `InputLayer`.

When using `InputLayer` with the Keras Sequential model, it can be skipped
by moving the `input_shape` parameter to the first layer after the
`InputLayer`.

This class can create placeholders for `tf.Tensors`, `tf.SparseTensors`, and
`tf.RaggedTensors` by choosing `sparse=True` or `ragged=True`. Note that
`sparse` and `ragged` can't be configured to `True` at the same time.
Usage:

```python
# With explicit InputLayer.
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(4,)),
  tf.keras.layers.Dense(8)])
model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
model.fit(np.zeros((10, 4)),
          np.ones((10, 8)))

# Without InputLayer and let the first layer to have the input_shape.
# Keras will add a input for the model behind the scene.
model = tf.keras.Sequential([
  tf.keras.layers.Dense(8, input_shape=(4,))])
model.compile(tf.keras.optimizers.RMSprop(0.001), loss='mse')
model.fit(np.zeros((10, 4)),
          np.ones((10, 8)))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_shape`<a id="input_shape"></a>
</td>
<td>
Shape tuple (not including the batch axis), or
`TensorShape` instance (not including the batch axis).
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Optional input batch size (integer or `None`).
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Optional datatype of the input. When not provided, the Keras
default `float` type will be used.
</td>
</tr><tr>
<td>
`input_tensor`<a id="input_tensor"></a>
</td>
<td>
Optional tensor to use as layer input. If set, the layer
will use the <a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> of this tensor rather
than creating a new placeholder tensor.
</td>
</tr><tr>
<td>
`sparse`<a id="sparse"></a>
</td>
<td>
Boolean, whether the placeholder created is meant to be sparse.
Defaults to `False`.
</td>
</tr><tr>
<td>
`ragged`<a id="ragged"></a>
</td>
<td>
Boolean, whether the placeholder created is meant to be ragged.
In this case, values of `None` in the `shape` argument represent
ragged dimensions. For more information about <a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>, see
[this guide](https://www.tensorflow.org/guide/ragged_tensor).
Defaults to `False`.
</td>
</tr><tr>
<td>
`type_spec`<a id="type_spec"></a>
</td>
<td>
A <a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> object to create Input from. This
<a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> represents the entire batch. When provided, all other
args except name must be `None`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name of the layer (string).
</td>
</tr>
</table>



