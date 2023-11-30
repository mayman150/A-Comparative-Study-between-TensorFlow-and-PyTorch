description: Adjust jpeg encoding quality of an image.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.adjust_jpeg_quality" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.adjust_jpeg_quality

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Adjust jpeg encoding quality of an image.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.adjust_jpeg_quality`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.adjust_jpeg_quality(
    image, jpeg_quality, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a convenience method that converts an image to uint8 representation,
encodes it to jpeg with `jpeg_quality`, decodes it, and then converts back
to the original data type.

`jpeg_quality` must be in the interval `[0, 100]`.

#### Usage Examples:



```
>>> x = [[[0.01, 0.02, 0.03],
...       [0.04, 0.05, 0.06]],
...      [[0.07, 0.08, 0.09],
...       [0.10, 0.11, 0.12]]]
>>> x_jpeg = tf.image.adjust_jpeg_quality(x, 75)
>>> x_jpeg.numpy()
array([[[0.00392157, 0.01960784, 0.03137255],
        [0.02745098, 0.04313726, 0.05490196]],
       [[0.05882353, 0.07450981, 0.08627451],
        [0.08235294, 0.09803922, 0.10980393]]], dtype=float32)
```

Note that floating point values are expected to have values in the range
[0,1) and values outside this range are clipped.

```
>>> x = [[[1.0, 2.0, 3.0],
...       [4.0, 5.0, 6.0]],
...     [[7.0, 8.0, 9.0],
...       [10.0, 11.0, 12.0]]]
>>> tf.image.adjust_jpeg_quality(x, 75)
<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
array([[[1., 1., 1.],
        [1., 1., 1.]],
       [[1., 1., 1.],
        [1., 1., 1.]]], dtype=float32)>
```

Note that `jpeg_quality` 100 is still lossy compresson.

```
>>> x = tf.constant([[[1, 2, 3],
...                   [4, 5, 6]],
...                  [[7, 8, 9],
...                   [10, 11, 12]]], dtype=tf.uint8)
>>> tf.image.adjust_jpeg_quality(x, 100)
<tf.Tensor: shape(2, 2, 3), dtype=uint8, numpy=
array([[[ 0,  1,  3],
        [ 3,  4,  6]],
       [[ 6,  7,  9],
        [ 9, 10, 12]]], dtype=uint8)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`image`<a id="image"></a>
</td>
<td>
3D image. The size of the last dimension must be None, 1 or 3.
</td>
</tr><tr>
<td>
`jpeg_quality`<a id="jpeg_quality"></a>
</td>
<td>
Python int or Tensor of type int32. jpeg encoding quality.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Adjusted image, same shape and DType as `image`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`<a id="InvalidArgumentError"></a>
</td>
<td>
quality must be in [0,100]
</td>
</tr><tr>
<td>
`InvalidArgumentError`<a id="InvalidArgumentError"></a>
</td>
<td>
image must have 1 or 3 channels
</td>
</tr>
</table>

