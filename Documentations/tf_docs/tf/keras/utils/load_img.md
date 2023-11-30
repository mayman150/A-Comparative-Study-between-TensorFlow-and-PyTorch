description: Loads an image into PIL format.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.load_img" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.load_img

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/image_utils.py#L364-L480">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads an image into PIL format.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.preprocessing.image.load_img`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.load_img(
    path,
    grayscale=False,
    color_mode=&#x27;rgb&#x27;,
    target_size=None,
    interpolation=&#x27;nearest&#x27;,
    keep_aspect_ratio=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Usage:



```python
image = tf.keras.utils.load_img(image_path)
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`<a id="path"></a>
</td>
<td>
Path to image file.
</td>
</tr><tr>
<td>
`grayscale`<a id="grayscale"></a>
</td>
<td>
DEPRECATED use `color_mode="grayscale"`.
</td>
</tr><tr>
<td>
`color_mode`<a id="color_mode"></a>
</td>
<td>
One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
The desired image format.
</td>
</tr><tr>
<td>
`target_size`<a id="target_size"></a>
</td>
<td>
Either `None` (default to original size) or tuple of ints
`(img_height, img_width)`.
</td>
</tr><tr>
<td>
`interpolation`<a id="interpolation"></a>
</td>
<td>
Interpolation method used to resample the image if the
target size is different from that of the loaded image. Supported
methods are `"nearest"`, `"bilinear"`, and `"bicubic"`. If PIL version
1.1.3 or newer is installed, `"lanczos"` is also supported. If PIL
version 3.4.0 or newer is installed, `"box"` and `"hamming"` are also
supported. By default, `"nearest"` is used.
</td>
</tr><tr>
<td>
`keep_aspect_ratio`<a id="keep_aspect_ratio"></a>
</td>
<td>
Boolean, whether to resize images to a target
size without aspect ratio distortion. The image is cropped in
the center with target aspect ratio before resizing.
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
if interpolation method is not supported.
</td>
</tr>
</table>

