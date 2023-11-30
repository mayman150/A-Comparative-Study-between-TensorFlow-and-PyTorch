description: A preprocessing layer which randomly varies image width during training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RandomWidth" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.RandomWidth

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/image_preprocessing.py#L1632-L1750">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which randomly varies image width during training.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.RandomWidth`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RandomWidth(
    factor, interpolation=&#x27;bilinear&#x27;, seed=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will randomly adjusts the width of a batch of images of a
batch of images by a random factor. The input should be a 3D (unbatched) or
4D (batched) tensor in the `"channels_last"` image data format. Input pixel
values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and of integer or
floating point dtype. By default, the layer will output floats.

By default, this layer is inactive during inference.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`factor`<a id="factor"></a>
</td>
<td>
A positive float (fraction of original width),
or a tuple of size 2 representing lower and upper bound
for resizing horizontally. When represented as a single float,
this value is used for both the upper and
lower bound. For instance, `factor=(0.2, 0.3)`
results in an output with
width changed by a random amount in the range `[20%, 30%]`.
`factor=(-0.2, 0.3)` results in an output with width changed
by a random amount in the range `[-20%, +30%]`.
`factor=0.2` results in an output with width changed
by a random amount in the range `[-20%, +20%]`.
</td>
</tr><tr>
<td>
`interpolation`<a id="interpolation"></a>
</td>
<td>
String, the interpolation method.
Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
`"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
Defaults to `bilinear`.
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

`(..., height, random_width, channels)`.
</td>
</tr>
</table>



