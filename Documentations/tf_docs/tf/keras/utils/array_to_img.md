description: Converts a 3D Numpy array to a PIL Image instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.array_to_img" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.array_to_img

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/image_utils.py#L201-L279">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a 3D Numpy array to a PIL Image instance.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.preprocessing.image.array_to_img`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.array_to_img(
    x, data_format=None, scale=True, dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Usage:



```python
from PIL import Image
img = np.random.random(size=(100, 100, 3))
pil_img = tf.keras.utils.array_to_img(img)
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
Input data, in any form that can be converted to a Numpy array.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
Image data format, can be either `"channels_first"` or
`"channels_last"`. None means the global
setting <a href="../../../tf/keras/backend/image_data_format.md"><code>tf.keras.backend.image_data_format()</code></a> is used (unless you
changed it, it uses `"channels_last"`). Defaults to `None`.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
Whether to rescale the image such that minimum and maximum values
are 0 and 255 respectively. Defaults to `True`.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Dtype to use. None makes the global setting
<a href="../../../tf/keras/backend/floatx.md"><code>tf.keras.backend.floatx()</code></a> to be used (unless you changed it, it
uses `"float32"`). Defaults to `None`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A PIL Image instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ImportError`<a id="ImportError"></a>
</td>
<td>
if PIL is not available.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if invalid `x` or `data_format` is passed.
</td>
</tr>
</table>

