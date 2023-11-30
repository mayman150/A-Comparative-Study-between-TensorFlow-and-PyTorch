description: Layer that averages a list of inputs element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Average" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Average

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/merging/average.py#L24-L58">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Layer that averages a list of inputs element-wise.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Average(
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

It takes as input a list of tensors, all of the same shape, and returns
a single tensor (also of the same shape).

#### Example:



```
>>> x1 = np.ones((2, 2))
>>> x2 = np.zeros((2, 2))
>>> y = tf.keras.layers.Average()([x1, x2])
>>> y.numpy().tolist()
[[0.5, 0.5], [0.5, 0.5]]
```

Usage in a functional model:

```
>>> input1 = tf.keras.layers.Input(shape=(16,))
>>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
>>> input2 = tf.keras.layers.Input(shape=(32,))
>>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
>>> avg = tf.keras.layers.Average()([x1, x2])
>>> out = tf.keras.layers.Dense(4)(avg)
>>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If there is a shape mismatch between the inputs and the shapes
cannot be broadcasted to match.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
standard layer keyword arguments.
</td>
</tr>
</table>



