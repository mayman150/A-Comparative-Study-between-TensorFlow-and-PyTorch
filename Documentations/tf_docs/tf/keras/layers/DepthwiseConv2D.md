description: Depthwise 2D convolution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.DepthwiseConv2D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convolution_op"/>
</div>

# tf.keras.layers.DepthwiseConv2D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/convolutional/depthwise_conv2d.py#L27-L209">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Depthwise 2D convolution.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.DepthwiseConv2D(
    kernel_size,
    strides=(1, 1),
    padding=&#x27;valid&#x27;,
    depth_multiplier=1,
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    depthwise_initializer=&#x27;glorot_uniform&#x27;,
    bias_initializer=&#x27;zeros&#x27;,
    depthwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    bias_constraint=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Depthwise convolution is a type of convolution in which each input channel
is convolved with a different kernel (called a depthwise kernel). You can
understand depthwise convolution as the first step in a depthwise separable
convolution.

It is implemented via the following steps:

- Split the input into individual channels.
- Convolve each channel with an individual depthwise kernel with
  `depth_multiplier` output channels.
- Concatenate the convolved outputs along the channels axis.

Unlike a regular 2D convolution, depthwise convolution does not mix
information across different input channels.

The `depth_multiplier` argument determines how many filter are applied to
one input channel. As such, it controls the amount of output channels that
are generated per input channel in the depthwise step.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`kernel_size`<a id="kernel_size"></a>
</td>
<td>
An integer or tuple/list of 2 integers, specifying the height
and width of the 2D convolution window. Can be a single integer to
specify the same value for all spatial dimensions.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
An integer or tuple/list of 2 integers, specifying the strides of
the convolution along the height and width. Can be a single integer to
specify the same value for all spatial dimensions. Current
implementation only supports equal length strides in row and
column dimensions. Specifying any stride value != 1 is incompatible
with specifying any `dilation_rate` value !=1.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
one of `'valid'` or `'same'` (case-insensitive). `"valid"` means
no padding. `"same"` results in padding with zeros evenly to the
left/right or up/down of the input such that output has the same
height/width dimension as the input.
</td>
</tr><tr>
<td>
`depth_multiplier`<a id="depth_multiplier"></a>
</td>
<td>
The number of depthwise convolution output channels for
each input channel. The total number of depthwise convolution output
channels will be equal to `filters_in * depth_multiplier`.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
A string, one of `channels_last` (default) or
`channels_first`. The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape `(batch_size, height,
width, channels)` while `channels_first` corresponds to inputs with
shape `(batch_size, channels, height, width)`. When unspecified, uses
`image_data_format` value found in your Keras config file at
 `~/.keras/keras.json` (if exists) else 'channels_last'.
Defaults to 'channels_last'.
</td>
</tr><tr>
<td>
`dilation_rate`<a id="dilation_rate"></a>
</td>
<td>
An integer or tuple/list of 2 integers, specifying the
dilation rate to use for dilated convolution. Currently, specifying any
`dilation_rate` value != 1 is incompatible with specifying any `strides`
value != 1.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
Activation function to use. If you don't specify anything, no
activation is applied (see <a href="../../../tf/keras/activations.md"><code>keras.activations</code></a>).
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
`depthwise_initializer`<a id="depthwise_initializer"></a>
</td>
<td>
Initializer for the depthwise kernel matrix (see
<a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). If None, the default initializer
('glorot_uniform') will be used.
</td>
</tr><tr>
<td>
`bias_initializer`<a id="bias_initializer"></a>
</td>
<td>
Initializer for the bias vector (see
<a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). If None, the default initializer ('zeros') will
be used.
</td>
</tr><tr>
<td>
`depthwise_regularizer`<a id="depthwise_regularizer"></a>
</td>
<td>
Regularizer function applied to the depthwise
kernel matrix (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`bias_regularizer`<a id="bias_regularizer"></a>
</td>
<td>
Regularizer function applied to the bias vector (see
<a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`activity_regularizer`<a id="activity_regularizer"></a>
</td>
<td>
Regularizer function applied to the output of the
layer (its 'activation') (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`depthwise_constraint`<a id="depthwise_constraint"></a>
</td>
<td>
Constraint function applied to the depthwise kernel
matrix (see <a href="../../../tf/keras/constraints.md"><code>keras.constraints</code></a>).
</td>
</tr><tr>
<td>
`bias_constraint`<a id="bias_constraint"></a>
</td>
<td>
Constraint function applied to the bias vector (see
<a href="../../../tf/keras/constraints.md"><code>keras.constraints</code></a>).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
4D tensor with shape: `[batch_size, channels, rows, cols]` if
  data_format='channels_first'
or 4D tensor with shape: `[batch_size, rows, cols, channels]` if
  data_format='channels_last'.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
4D tensor with shape: `[batch_size, channels * depth_multiplier, new_rows,
new_cols]` if `data_format='channels_first'`
or 4D tensor with shape: `[batch_size,
new_rows, new_cols, channels * depth_multiplier]` if
`data_format='channels_last'`. `rows` and `cols` values might have
changed due to padding.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of rank 4 representing
`activation(depthwiseconv2d(inputs, kernel) + bias)`.
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



## Methods

<h3 id="convolution_op"><code>convolution_op</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/convolutional/base_conv.py#L254-L270">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convolution_op(
    inputs, kernel
)
</code></pre>






