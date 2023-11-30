description: Performs a random brightness shift.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.random_brightness" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.image.random_brightness

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L2438-L2470">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Performs a random brightness shift.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.random_brightness(
    x, brightness_range, scale=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/image/random_brightness.md"><code>tf.keras.preprocessing.image.random_brightness</code></a> does not
operate on tensors and is not recommended for new code. Prefer
<a href="../../../../tf/keras/layers/RandomBrightness.md"><code>tf.keras.layers.RandomBrightness</code></a> which provides equivalent functionality
as a preprocessing layer. For more information, see the tutorial for
[augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
the [preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
Input tensor. Must be 3D.
</td>
</tr><tr>
<td>
`brightness_range`<a id="brightness_range"></a>
</td>
<td>
Tuple of floats; brightness range.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
Whether to rescale the image such that minimum and maximum values
are 0 and 255 respectively. Default: True.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Numpy image tensor.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if `brightness_range` isn't a tuple.
</td>
</tr>

</table>

