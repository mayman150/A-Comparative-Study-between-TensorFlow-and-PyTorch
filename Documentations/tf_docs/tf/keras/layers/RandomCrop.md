description: A preprocessing layer which randomly crops images during training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RandomCrop" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.RandomCrop

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/image_preprocessing.py#L232-L325">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which randomly crops images during training.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.RandomCrop`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RandomCrop(
    height, width, seed=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

During training, this layer will randomly choose a location to crop images
down to a target size. The layer will crop all the images in the same batch
to the same cropping location.

At inference time, and during training if an input image is smaller than the
target size, the input will be resized and cropped so as to return the
largest possible window in the image that matches the target aspect ratio.
If you need to apply random cropping at inference time, set `training` to
True when calling the layer.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
of integer or floating point dtype. By default, the layer will output
floats.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>

<tr>
<td>
`3D`<a id="3D"></a>
</td>
<td>
`unbatched) or 4D (batched) tensor with shape`

`(..., height, width, channels)`, in `"channels_last"` format.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>

<tr>
<td>
`3D`<a id="3D"></a>
</td>
<td>
`unbatched) or 4D (batched) tensor with shape`

`(..., target_height, target_width, channels)`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`height`<a id="height"></a>
</td>
<td>
Integer, the height of the output shape.
</td>
</tr><tr>
<td>
`width`<a id="width"></a>
</td>
<td>
Integer, the width of the output shape.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Integer. Used to create a random seed.
</td>
</tr>
</table>



