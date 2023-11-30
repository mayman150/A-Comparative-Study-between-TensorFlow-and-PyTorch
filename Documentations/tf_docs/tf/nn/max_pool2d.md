description: Performs max pooling on 2D spatial data such as images.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.max_pool2d" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.max_pool2d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Performs max pooling on 2D spatial data such as images.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nn.max_pool2d`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.max_pool2d(
    input, ksize, strides, padding, data_format=&#x27;NHWC&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a more specific version of <a href="../../tf/nn/max_pool.md"><code>tf.nn.max_pool</code></a> where the input tensor
is 4D, representing 2D spatial data such as images. Using these APIs are
equivalent

Downsamples the input images along theirs spatial dimensions (height and
width) by taking its maximum over an input window defined by `ksize`.
The window is shifted by `strides` along each dimension.

For example, for `strides=(2, 2)` and `padding=VALID` windows that extend
outside of the input are not included in the output:

```
>>> x = tf.constant([[1., 2., 3., 4.],
...                  [5., 6., 7., 8.],
...                  [9., 10., 11., 12.]])
>>> # Add the `batch` and `channels` dimensions.
>>> x = x[tf.newaxis, :, :, tf.newaxis]
>>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
...                           padding="VALID")
>>> result[0, :, :, 0]
<tf.Tensor: shape=(1, 2), dtype=float32, numpy=
array([[6., 8.]], dtype=float32)>
```

With `padding=SAME`, we get:

```
>>> x = tf.constant([[1., 2., 3., 4.],
...                  [5., 6., 7., 8.],
...                  [9., 10., 11., 12.]])
>>> x = x[tf.newaxis, :, :, tf.newaxis]
>>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
...                           padding='SAME')
>>> result[0, :, :, 0]
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 6., 8.],
       [10.,12.]], dtype=float32)>
```

We can also specify padding explicitly. The following example adds width-1
padding on all sides (top, bottom, left, right):

```
>>> x = tf.constant([[1., 2., 3., 4.],
...                  [5., 6., 7., 8.],
...                  [9., 10., 11., 12.]])
>>> x = x[tf.newaxis, :, :, tf.newaxis]
>>> result = tf.nn.max_pool2d(x, ksize=(2, 2), strides=(2, 2),
...                           padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
>>> result[0, :, :, 0]
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[ 1., 3., 4.],
       [ 9., 11., 12.]], dtype=float32)>
```

For more examples and detail, see <a href="../../tf/nn/max_pool.md"><code>tf.nn.max_pool</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A 4-D `Tensor` of the format specified by `data_format`.
</td>
</tr><tr>
<td>
`ksize`<a id="ksize"></a>
</td>
<td>
An int or list of `ints` that has length `1`, `2` or `4`. The size of
the window for each dimension of the input tensor. If only one integer is
specified, then we apply the same window for all 4 dims. If two are
provided then we use those for H, W dimensions and keep N, C dimension
window size = 1.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
An int or list of `ints` that has length `1`, `2` or `4`. The
stride of the sliding window for each dimension of the input tensor. If
only one integer is specified, we apply the same stride to all 4 dims. If
two are provided we use those for the H, W dimensions and keep N, C of
stride = 1.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
Either the `string` `"SAME"` or `"VALID"` indicating the type of
padding algorithm to use, or a list indicating the explicit paddings at
the start and end of each dimension. See
[here](https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2)
  for more information. When explicit padding is used and data_format is
  `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
  [pad_left, pad_right], [0, 0]]`. When explicit padding used and
  data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
  [pad_top, pad_bottom], [pad_left, pad_right]]`. When using explicit
  padding, the size of the paddings cannot be greater than the sliding
  window size.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of format specified by `data_format`.
The max pooled output tensor.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If explicit padding is used with data_format='NCHW_VECT_C'.
</td>
</tr>
</table>

