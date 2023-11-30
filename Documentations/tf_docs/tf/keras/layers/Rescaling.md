description: A preprocessing layer which rescales input values to a new range.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Rescaling" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Rescaling

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/image_preprocessing.py#L328-L386">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which rescales input values to a new range.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Rescaling`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Rescaling(
    scale, offset=0.0, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer rescales every value of an input (often an image) by multiplying
by `scale` and adding `offset`.

#### For instance:



1. To rescale an input in the `[0, 255]` range
to be in the `[0, 1]` range, you would pass `scale=1./255`.

2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
you would pass `scale=1./127.5, offset=-1`.

The rescaling is applied both during training and inference. Inputs can be
of integer or floating point dtype, and by default the layer will output
floats.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Arbitrary.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
Same as input.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
Float, the scale to apply to the inputs.
</td>
</tr><tr>
<td>
`offset`<a id="offset"></a>
</td>
<td>
Float, the offset to apply to the inputs.
</td>
</tr>
</table>



