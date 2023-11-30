description: Zero-padding layer for 3D data (spatial or spatio-temporal).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.ZeroPadding3D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.ZeroPadding3D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/reshaping/zero_padding3d.py#L29-L164">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Zero-padding layer for 3D data (spatial or spatio-temporal).

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.ZeroPadding3D(
    padding=(1, 1, 1), data_format=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Examples:



```
>>> input_shape = (1, 1, 2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> y = tf.keras.layers.ZeroPadding3D(padding=2)(x)
>>> print(y.shape)
(1, 5, 6, 6, 3)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
- If int: the same symmetric padding
  is applied to height and width.
- If tuple of 3 ints:
  interpreted as two different
  symmetric padding values for height and width:
  `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
- If tuple of 3 tuples of 2 ints:
  interpreted as
  `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
    right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
A string,
one of `channels_last` (default) or `channels_first`.
The ordering of the dimensions in the inputs.
`channels_last` corresponds to inputs with shape
`(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
while `channels_first` corresponds to inputs with shape
`(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
When unspecified, uses
`image_data_format` value found in your Keras config file at
 `~/.keras/keras.json` (if exists) else 'channels_last'.
Defaults to 'channels_last'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch_size, first_axis_to_pad, second_axis_to_pad,
    third_axis_to_pad, depth)`
- If `data_format` is `"channels_first"`:
    `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
    third_axis_to_pad)`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch_size, first_padded_axis, second_padded_axis,
    third_axis_to_pad, depth)`
- If `data_format` is `"channels_first"`:
    `(batch_size, depth, first_padded_axis, second_padded_axis,
      third_axis_to_pad)`
</td>
</tr>

</table>



