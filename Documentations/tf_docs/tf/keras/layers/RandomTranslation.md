description: A preprocessing layer which randomly translates images during training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.RandomTranslation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.RandomTranslation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/image_preprocessing.py#L499-L683">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which randomly translates images during training.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.RandomTranslation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.RandomTranslation(
    height_factor,
    width_factor,
    fill_mode=&#x27;reflect&#x27;,
    interpolation=&#x27;bilinear&#x27;,
    seed=None,
    fill_value=0.0,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer will apply random translations to each image during training,
filling empty space according to `fill_mode`.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
of integer or floating point dtype. By default, the layer will output
floats.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`height_factor`<a id="height_factor"></a>
</td>
<td>
a float represented as fraction of value, or a tuple of
size 2 representing lower and upper bound for shifting vertically. A
negative value means shifting image up, while a positive value means
shifting image down. When represented as a single positive float, this
value is used for both the upper and lower bound. For instance,
`height_factor=(-0.2, 0.3)` results in an output shifted by a random
amount in the range `[-20%, +30%]`.  `height_factor=0.2` results in an
output height shifted by a random amount in the range `[-20%, +20%]`.
</td>
</tr><tr>
<td>
`width_factor`<a id="width_factor"></a>
</td>
<td>
a float represented as fraction of value, or a tuple of size
2 representing lower and upper bound for shifting horizontally. A
negative value means shifting image left, while a positive value means
shifting image right. When represented as a single positive float,
this value is used for both the upper and lower bound. For instance,
`width_factor=(-0.2, 0.3)` results in an output shifted left by 20%,
and shifted right by 30%. `width_factor=0.2` results
in an output height shifted left or right by 20%.
</td>
</tr><tr>
<td>
`fill_mode`<a id="fill_mode"></a>
</td>
<td>
Points outside the boundaries of the input are filled according
to the given mode
(one of `{"constant", "reflect", "wrap", "nearest"}`).
- *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
    reflecting about the edge of the last pixel.
- *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
    filling all values beyond the edge with the same constant value
    k = 0.
- *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
    wrapping around to the opposite edge.
- *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
    the nearest pixel.
</td>
</tr><tr>
<td>
`interpolation`<a id="interpolation"></a>
</td>
<td>
Interpolation mode. Supported values: `"nearest"`,
`"bilinear"`.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Integer. Used to create a random seed.
</td>
</tr><tr>
<td>
`fill_value`<a id="fill_value"></a>
</td>
<td>
a float represents the value to be filled outside the
boundaries when `fill_mode="constant"`.
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

`(..., height, width, channels)`,  in `"channels_last"` format.
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

`(..., height, width, channels)`,  in `"channels_last"` format.
</td>
</tr>
</table>



