description: Spatial 3D version of Dropout.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.SpatialDropout3D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.SpatialDropout3D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/regularization/spatial_dropout3d.py#L28-L79">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Spatial 3D version of Dropout.

Inherits From: [`Dropout`](../../../tf/keras/layers/Dropout.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.SpatialDropout3D(
    rate, data_format=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This version performs the same function as Dropout, however, it drops
entire 3D feature maps instead of individual elements. If adjacent voxels
within feature maps are strongly correlated (as is normally the case in
early convolution layers) then regular dropout will not regularize the
activations and will otherwise just result in an effective learning rate
decrease. In this case, SpatialDropout3D will help promote independence
between feature maps and should be used instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`rate`<a id="rate"></a>
</td>
<td>
Float between 0 and 1. Fraction of the input units to drop.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
'channels_first' or 'channels_last'. In 'channels_first'
mode, the channels dimension (the depth) is at index 1, in
'channels_last' mode is it at index 4. When unspecified, uses
`image_data_format` value found in your Keras config file at
`~/.keras/keras.json` (if exists) else 'channels_last'.
Defaults to 'channels_last'.
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
A 5D tensor.
</td>
</tr><tr>
<td>
`training`<a id="training"></a>
</td>
<td>
Python boolean indicating whether the layer should behave in
training mode (adding dropout) or in inference mode (doing nothing).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
5D tensor with shape: `(samples, channels, dim1, dim2, dim3)` if
  data_format='channels_first'
or 5D tensor with shape: `(samples, dim1, dim2, dim3, channels)` if
  data_format='channels_last'.
</td>
</tr>

</table>


Output shape: Same as input.
References: - [Efficient Object Localization Using Convolutional
    Networks](https://arxiv.org/abs/1411.4280)

