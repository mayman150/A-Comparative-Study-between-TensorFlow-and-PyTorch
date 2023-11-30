description: Unit normalization layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.UnitNormalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.UnitNormalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/normalization/unit_normalization.py#L27-L75">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Unit normalization layer.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.UnitNormalization(
    axis=-1, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Normalize a batch of inputs so that each input in the batch has a L2 norm
equal to 1 (across the axes specified in `axis`).

#### Example:



```
>>> data = tf.constant(np.arange(6).reshape(2, 3), dtype=tf.float32)
>>> normalized_data = tf.keras.layers.UnitNormalization()(data)
>>> print(tf.reduce_sum(normalized_data[0, :] ** 2).numpy())
1.0
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
Integer or list/tuple. The axis or axes to normalize across.
Typically, this is the features axis or axes. The left-out axes are
typically the batch axis or axes. `-1` is the last dimension
in the input. Defaults to `-1`.
</td>
</tr>
</table>



