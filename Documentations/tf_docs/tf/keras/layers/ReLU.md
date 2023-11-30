description: Rectified Linear Unit activation function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.ReLU" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.ReLU

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/activation/relu.py#L26-L124">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Rectified Linear Unit activation function.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.ReLU(
    max_value=None, negative_slope=0.0, threshold=0.0, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

With default values, it returns element-wise `max(x, 0)`.

Otherwise, it follows:

```
    f(x) = max_value if x >= max_value
    f(x) = x if threshold <= x < max_value
    f(x) = negative_slope * (x - threshold) otherwise
```

#### Usage:



```
>>> layer = tf.keras.layers.ReLU()
>>> output = layer([-3.0, -1.0, 0.0, 2.0])
>>> list(output.numpy())
[0.0, 0.0, 0.0, 2.0]
>>> layer = tf.keras.layers.ReLU(max_value=1.0)
>>> output = layer([-3.0, -1.0, 0.0, 2.0])
>>> list(output.numpy())
[0.0, 0.0, 0.0, 1.0]
>>> layer = tf.keras.layers.ReLU(negative_slope=1.0)
>>> output = layer([-3.0, -1.0, 0.0, 2.0])
>>> list(output.numpy())
[-3.0, -1.0, 0.0, 2.0]
>>> layer = tf.keras.layers.ReLU(threshold=1.5)
>>> output = layer([-3.0, -1.0, 1.0, 2.0])
>>> list(output.numpy())
[0.0, 0.0, 0.0, 2.0]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the batch axis)
when using this layer as the first layer in a model.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Same shape as the input.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`max_value`<a id="max_value"></a>
</td>
<td>
Float >= 0. Maximum activation value. None means unlimited.
Defaults to `None`.
</td>
</tr><tr>
<td>
`negative_slope`<a id="negative_slope"></a>
</td>
<td>
Float >= 0. Negative slope coefficient.
Defaults to `0.`.
</td>
</tr><tr>
<td>
`threshold`<a id="threshold"></a>
</td>
<td>
Float >= 0. Threshold value for thresholded activation.
Defaults to `0.`.
</td>
</tr>
</table>



