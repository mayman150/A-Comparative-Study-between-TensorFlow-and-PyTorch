description: Layer that multiplies (element-wise) a list of inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Multiply" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Multiply

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/merging/multiply.py#L24-L51">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Layer that multiplies (element-wise) a list of inputs.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Multiply(
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

It takes as input a list of tensors, all of the same shape, and returns
a single tensor (also of the same shape).

```
>>> tf.keras.layers.Multiply()([np.arange(5).reshape(5, 1),
...                             np.arange(5, 10).reshape(5, 1)])
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
array([[ 0],
     [ 6],
     [14],
     [24],
     [36]])>
```

```
>>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
>>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
>>> multiplied = tf.keras.layers.Multiply()([x1, x2])
>>> multiplied.shape
TensorShape([5, 8])
```

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



