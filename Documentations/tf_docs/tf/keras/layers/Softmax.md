description: Softmax activation function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Softmax" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Softmax

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/activation/softmax.py#L48-L117">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Softmax activation function.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Softmax(
    axis=-1, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example without mask:



```
>>> inp = np.asarray([[1., 2., 1.]])
>>> layer = tf.keras.layers.Softmax()
>>> layer(inp).numpy()
array([[0.21194157, 0.5761169 , 0.21194157]], dtype=float32)
>>> mask = np.asarray([[True, False, True]], dtype=bool)
>>> layer(inp, mask).numpy()
array([[0.5, 0. , 0.5]], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
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
`axis`<a id="axis"></a>
</td>
<td>
Integer, or list of Integers, axis along which the softmax
normalization is applied.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call arguments</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
The inputs, or logits to the softmax layer.
</td>
</tr><tr>
<td>
`mask`<a id="mask"></a>
</td>
<td>
A boolean mask of the same shape as `inputs`. The mask
specifies 1 to keep and 0 to mask. Defaults to `None`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Softmaxed output with the same shape as `inputs`.
</td>
</tr>

</table>



