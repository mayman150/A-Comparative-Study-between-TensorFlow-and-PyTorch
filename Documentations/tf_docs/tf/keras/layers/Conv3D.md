description: 3D convolution layer (e.g. spatial convolution over volumes).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Conv3D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convolution_op"/>
</div>

# tf.keras.layers.Conv3D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/convolutional/conv3d.py#L29-L182">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



3D convolution layer (e.g. spatial convolution over volumes).

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.Convolution3D`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Conv3D(
    filters,
    kernel_size,
    strides=(1, 1, 1),
    padding=&#x27;valid&#x27;,
    data_format=None,
    dilation_rate=(1, 1, 1),
    groups=1,
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

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs. Finally, if
`activation` is not `None`, it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers or `None`, does not include the sample axis),
e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
with a single channel,
in `data_format="channels_last"`.

#### Examples:



```
>>> # The inputs are 28x28x28 volumes with a single channel, and the
>>> # batch size is 4
>>> input_shape =(4, 28, 28, 28, 1)
>>> x = tf.random.normal(input_shape)
>>> y = tf.keras.layers.Conv3D(
... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
>>> print(y.shape)
(4, 26, 26, 26, 2)
```

```
>>> # With extended batch shape [4, 7], e.g. a batch of 4 videos of
>>> # 3D frames, with 7 frames per video.
>>> input_shape = (4, 7, 28, 28, 28, 1)
>>> x = tf.random.normal(input_shape)
>>> y = tf.keras.layers.Conv3D(
... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
>>> print(y.shape)
(4, 7, 26, 26, 26, 2)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filters`<a id="filters"></a>
</td>
<td>
Integer, the dimensionality of the output space (i.e. the number
of output filters in the convolution).
</td>
</tr><tr>
<td>
`kernel_size`<a id="kernel_size"></a>
</td>
<td>
An integer or tuple/list of 3 integers, specifying the depth,
height and width of the 3D convolution window. Can be a single integer
to specify the same value for all spatial dimensions.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
An integer or tuple/list of 3 integers, specifying the strides of
the convolution along each spatial dimension. Can be a single integer to
specify the same value for all spatial dimensions. Specifying any stride
value != 1 is incompatible with specifying any `dilation_rate` value !=
1.
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
`data_format`<a id="data_format"></a>
</td>
<td>
A string, one of `channels_last` (default) or
`channels_first`.  The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape `batch_shape +
(spatial_dim1, spatial_dim2, spatial_dim3, channels)` while
`channels_first` corresponds to inputs with shape `batch_shape +
(channels, spatial_dim1, spatial_dim2, spatial_dim3)`. When unspecified,
uses `image_data_format` value found in your Keras config file at
`~/.keras/keras.json` (if exists) else 'channels_last'. Note that the
`channels_first` format is currently not supported by TensorFlow on CPU.
Defaults to 'channels_last'.
</td>
</tr><tr>
<td>
`dilation_rate`<a id="dilation_rate"></a>
</td>
<td>
an integer or tuple/list of 3 integers, specifying the
dilation rate to use for dilated convolution. Can be a single integer to
specify the same value for all spatial dimensions. Currently, specifying
any `dilation_rate` value != 1 is incompatible with specifying any
stride value != 1.
</td>
</tr><tr>
<td>
`groups`<a id="groups"></a>
</td>
<td>
A positive integer specifying the number of groups in which the
input is split along the channel axis. Each group is convolved
separately with `filters / groups` filters. The output is the
concatenation of all the `groups` results along the channel axis. Input
channels and `filters` must both be divisible by `groups`.
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
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Initializer for the `kernel` weights matrix (see
<a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). Defaults to 'glorot_uniform'.
</td>
</tr><tr>
<td>
`bias_initializer`<a id="bias_initializer"></a>
</td>
<td>
Initializer for the bias vector (see
<a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). Defaults to 'zeros'.
</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>
Regularizer function applied to the `kernel` weights
matrix (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
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
layer (its "activation") (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`kernel_constraint`<a id="kernel_constraint"></a>
</td>
<td>
Constraint function applied to the kernel matrix (see
<a href="../../../tf/keras/constraints.md"><code>keras.constraints</code></a>).
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
5+D tensor with shape: `batch_shape + (channels, conv_dim1, conv_dim2,
  conv_dim3)` if data_format='channels_first'
or 5+D tensor with shape: `batch_shape + (conv_dim1, conv_dim2, conv_dim3,
  channels)` if data_format='channels_last'.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
5+D tensor with shape: `batch_shape + (filters, new_conv_dim1,
  new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
or 5+D tensor with shape: `batch_shape + (new_conv_dim1, new_conv_dim2,
  new_conv_dim3, filters)` if data_format='channels_last'.
  `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
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
A tensor of rank 5+ representing
`activation(conv3d(inputs, kernel) + bias)`.
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
when both `strides > 1` and `dilation_rate > 1`.
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






