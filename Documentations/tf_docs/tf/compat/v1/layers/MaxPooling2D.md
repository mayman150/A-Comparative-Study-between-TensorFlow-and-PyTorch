description: Max pooling layer for 2D inputs (e.g. images).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.layers.MaxPooling2D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_updates_for"/>
</div>

# tf.compat.v1.layers.MaxPooling2D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/legacy_tf_layers/pooling.py#L503-L575">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Max pooling layer for 2D inputs (e.g. images).

Inherits From: [`MaxPooling2D`](../../../../tf/keras/layers/MaxPooling2D.md), [`Layer`](../../../../tf/compat/v1/layers/Layer.md), [`Layer`](../../../../tf/keras/layers/Layer.md), [`Module`](../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.layers.MaxPooling2D(
    pool_size,
    strides,
    padding=&#x27;valid&#x27;,
    data_format=&#x27;channels_last&#x27;,
    name=None,
    **kwargs
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is a legacy api that is only compatible with eager execution and
<a href="../../../../tf/function.md"><code>tf.function</code></a> if you combine it with
`tf.compat.v1.keras.utils.track_tf1_style_variables`

Please refer to [tf.layers model mapping section of the migration guide]
(https://www.tensorflow.org/guide/migrate/model_mapping)
to learn how to use your TensorFlow v1 model in TF2 with Keras.

The corresponding TensorFlow v2 layer is
<a href="../../../../tf/keras/layers/MaxPooling2D.md"><code>tf.keras.layers.MaxPooling2D</code></a>.


#### Structural Mapping to Native TF2

None of the supported arguments have changed name.

Before:

```python
 pooling = tf.compat.v1.layers.MaxPooling2D(pool_size=2, strides=2)
```

After:

```python
 pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
```

 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`pool_size`<a id="pool_size"></a>
</td>
<td>
An integer or tuple/list of 2 integers: (pool_height,
pool_width) specifying the size of the pooling window.
Can be a single integer to specify the same value for
all spatial dimensions.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
An integer or tuple/list of 2 integers,
specifying the strides of the pooling operation.
Can be a single integer to specify the same value for
all spatial dimensions.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
A string. The padding method, either 'valid' or 'same'.
Case-insensitive.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
A string. The ordering of the dimensions in the inputs.
`channels_last` (default) and `channels_first` are supported.
`channels_last` corresponds to inputs with shape
`(batch, height, width, channels)` while `channels_first` corresponds to
inputs with shape `(batch, channels, height, width)`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A string, the name of the layer.
</td>
</tr>
</table>






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`scope_name`<a id="scope_name"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="apply"><code>apply</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/legacy_tf_layers/base.py#L239-L240">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    *args, **kwargs
)
</code></pre>




<h3 id="get_losses_for"><code>get_losses_for</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/base_layer_v1.py#L1467-L1484">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_losses_for(
    inputs
)
</code></pre>

Retrieves losses relevant to a specific set of inputs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor or list/tuple of input tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of loss tensors of the layer that depend on `inputs`.
</td>
</tr>

</table>



<h3 id="get_updates_for"><code>get_updates_for</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/base_layer_v1.py#L1448-L1465">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_updates_for(
    inputs
)
</code></pre>

Retrieves updates relevant to a specific set of inputs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor or list/tuple of input tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of update ops of the layer that depend on `inputs`.
</td>
</tr>

</table>





