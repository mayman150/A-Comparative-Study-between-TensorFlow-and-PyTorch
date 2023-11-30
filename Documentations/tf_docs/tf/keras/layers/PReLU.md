description: Parametric Rectified Linear Unit.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.PReLU" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.PReLU

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/activation/prelu.py#L30-L124">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parametric Rectified Linear Unit.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.PReLU(
    alpha_initializer=&#x27;zeros&#x27;,
    alpha_regularizer=None,
    alpha_constraint=None,
    shared_axes=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### It follows:



```
    f(x) = alpha * x for x < 0
    f(x) = x for x >= 0
```

where `alpha` is a learned array with the same shape as x.

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
`alpha_initializer`<a id="alpha_initializer"></a>
</td>
<td>
Initializer function for the weights.
</td>
</tr><tr>
<td>
`alpha_regularizer`<a id="alpha_regularizer"></a>
</td>
<td>
Regularizer for the weights.
</td>
</tr><tr>
<td>
`alpha_constraint`<a id="alpha_constraint"></a>
</td>
<td>
Constraint for the weights.
</td>
</tr><tr>
<td>
`shared_axes`<a id="shared_axes"></a>
</td>
<td>
The axes along which to share learnable
parameters for the activation function.
For example, if the incoming feature maps
are from a 2D convolution
with output shape `(batch, height, width, channels)`,
and you wish to share parameters across space
so that each filter only has one set of parameters,
set `shared_axes=[1, 2]`.
</td>
</tr>
</table>



