description: Transposed convolution layer (sometimes called Deconvolution).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Conv1DTranspose" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convolution_op"/>
</div>

# tf.keras.layers.Conv1DTranspose

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/convolutional/conv1d_transpose.py#L33-L299">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Transposed convolution layer (sometimes called Deconvolution).

Inherits From: [`Conv1D`](../../../tf/keras/layers/Conv1D.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.Convolution1DTranspose`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Conv1DTranspose(
    filters,
    kernel_size,
    strides=1,
    padding=&#x27;valid&#x27;,
    output_padding=None,
    data_format=None,
    dilation_rate=1,
    activation=None,
    use_bias=True,
    kernel_initializer=&#x27;glorot_uniform&#x27;,
    bias_initializer=&#x27;zeros&#x27;,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers or `None`, does not include the sample axis),
e.g. `input_shape=(128, 3)` for data with 128 time steps and 3 channels.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filters`<a id="filters"></a>
</td>
<td>
Integer, the dimensionality of the output space
(i.e. the number of output filters in the convolution).
</td>
</tr><tr>
<td>
`kernel_size`<a id="kernel_size"></a>
</td>
<td>
An integer length of the 1D convolution window.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
An integer specifying the stride of the convolution along the
time dimension. Specifying a stride value != 1 is incompatible with
specifying a `dilation_rate` value != 1. Defaults to `1`.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
one of `"valid"` or `"same"` (case-insensitive).
`"valid"` means no padding. `"same"` results in padding with zeros
evenly to the left/right or up/down of the input such that output has
the same height/width dimension as the input.
</td>
</tr><tr>
<td>
`output_padding`<a id="output_padding"></a>
</td>
<td>
An integer specifying the amount of padding along
the time dimension of the output tensor.
The amount of output padding must be lower than the stride.
If set to `None` (default), the output shape is inferred.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
A string, one of `channels_last` (default) or
`channels_first`.  The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch_size, length, channels)` while `channels_first` corresponds to
inputs with shape `(batch_size, channels, length)`.
</td>
</tr><tr>
<td>
`dilation_rate`<a id="dilation_rate"></a>
</td>
<td>
an integer, specifying
the dilation rate to use for dilated convolution.
Currently, specifying a `dilation_rate` value != 1 is
incompatible with specifying a stride value != 1.
Also dilation rate larger than 1 is not currently supported.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
Activation function to use.
If you don't specify anything, no activation is applied
(see <a href="../../../tf/keras/activations.md"><code>keras.activations</code></a>).
</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>
Boolean, whether the layer uses a bias vector.
</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Initializer for the `kernel` weights matrix
(see <a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). Defaults to 'glorot_uniform'.
</td>
</tr><tr>
<td>
`bias_initializer`<a id="bias_initializer"></a>
</td>
<td>
Initializer for the bias vector
(see <a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). Defaults to 'zeros'.
</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>
Regularizer function applied to
the `kernel` weights matrix (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`bias_regularizer`<a id="bias_regularizer"></a>
</td>
<td>
Regularizer function applied to the bias vector
(see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`activity_regularizer`<a id="activity_regularizer"></a>
</td>
<td>
Regularizer function applied to
the output of the layer (its "activation") (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`kernel_constraint`<a id="kernel_constraint"></a>
</td>
<td>
Constraint function applied to the kernel matrix
(see <a href="../../../tf/keras/constraints.md"><code>keras.constraints</code></a>).
</td>
</tr><tr>
<td>
`bias_constraint`<a id="bias_constraint"></a>
</td>
<td>
Constraint function applied to the bias vector
(see <a href="../../../tf/keras/constraints.md"><code>keras.constraints</code></a>).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
3D tensor with shape:
`(batch_size, steps, channels)`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
3D tensor with shape:
`(batch_size, new_steps, filters)`
If `output_padding` is specified:
```
new_timesteps = ((timesteps - 1) * strides + kernel_size -
2 * padding + output_padding)
```
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of rank 3 representing
`activation(conv1dtranspose(inputs, kernel) + bias)`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `padding` is "causal".
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
when both `strides` > 1 and `dilation_rate` > 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
- [A guide to convolution arithmetic for deep learning](
  https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](
  https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
</td>
</tr>

</table>



## Methods

<h3 id="convolution_op"><code>convolution_op</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/convolutional/base_conv.py#L254-L270">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convolution_op(
    inputs, kernel
)
</code></pre>






