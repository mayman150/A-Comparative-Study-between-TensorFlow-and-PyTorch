description: Upsampling layer for 3D inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.UpSampling3D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.UpSampling3D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/reshaping/up_sampling3d.py#L29-L131">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Upsampling layer for 3D inputs.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.UpSampling3D(
    size=(2, 2, 2), data_format=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Repeats the 1st, 2nd and 3rd dimensions
of the data by `size[0]`, `size[1]` and `size[2]` respectively.

#### Examples:



```
>>> input_shape = (2, 1, 2, 1, 3)
>>> x = tf.constant(1, shape=input_shape)
>>> y = tf.keras.layers.UpSampling3D(size=2)(x)
>>> print(y.shape)
(2, 2, 4, 2, 3)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`size`<a id="size"></a>
</td>
<td>
Int, or tuple of 3 integers.
The upsampling factors for dim1, dim2 and dim3.
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
    `(batch_size, dim1, dim2, dim3, channels)`
- If `data_format` is `"channels_first"`:
    `(batch_size, channels, dim1, dim2, dim3)`
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
    `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3,
    channels)`
- If `data_format` is `"channels_first"`:
    `(batch_size, channels, upsampled_dim1, upsampled_dim2,
    upsampled_dim3)`
</td>
</tr>

</table>



