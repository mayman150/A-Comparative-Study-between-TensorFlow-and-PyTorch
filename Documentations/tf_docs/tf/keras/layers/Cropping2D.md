description: Cropping layer for 2D input (e.g. picture).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Cropping2D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Cropping2D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/reshaping/cropping2d.py#L28-L219">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Cropping layer for 2D input (e.g. picture).

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Cropping2D(
    cropping=((0, 0), (0, 0)), data_format=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

It crops along spatial dimensions, i.e. height and width.

#### Examples:



```
>>> input_shape = (2, 28, 28, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
>>> print(y.shape)
(2, 24, 20, 3)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cropping`<a id="cropping"></a>
</td>
<td>
Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
- If int: the same symmetric cropping
  is applied to height and width.
- If tuple of 2 ints:
  interpreted as two different
  symmetric cropping values for height and width:
  `(symmetric_height_crop, symmetric_width_crop)`.
- If tuple of 2 tuples of 2 ints:
  interpreted as
  `((top_crop, bottom_crop), (left_crop, right_crop))`
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
`(batch_size, height, width, channels)` while `channels_first`
corresponds to inputs with shape
`(batch_size, channels, height, width)`.
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
4D tensor with shape:
- If `data_format` is `"channels_last"`:
  `(batch_size, rows, cols, channels)`
- If `data_format` is `"channels_first"`:
  `(batch_size, channels, rows, cols)`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
4D tensor with shape:
- If `data_format` is `"channels_last"`:
  `(batch_size, cropped_rows, cropped_cols, channels)`
- If `data_format` is `"channels_first"`:
  `(batch_size, channels, cropped_rows, cropped_cols)`
</td>
</tr>

</table>



