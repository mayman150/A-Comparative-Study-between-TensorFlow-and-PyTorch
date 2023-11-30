description: Randomly changes jpeg encoding quality for inducing jpeg noise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.random_jpeg_quality" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.random_jpeg_quality

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/image_ops_impl.py">View source</a>



Randomly changes jpeg encoding quality for inducing jpeg noise.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.image.random_jpeg_quality`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.image.random_jpeg_quality(
    image, min_jpeg_quality, max_jpeg_quality, seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`min_jpeg_quality` must be in the interval `[0, 100]` and less than
`max_jpeg_quality`.
`max_jpeg_quality` must be in the interval `[0, 100]`.

#### Usage Example:



```
>>> x = tf.constant([[[1, 2, 3],
...                   [4, 5, 6]],
...                  [[7, 8, 9],
...                   [10, 11, 12]]], dtype=tf.uint8)
>>> tf.image.random_jpeg_quality(x, 75, 95)
<tf.Tensor: shape=(2, 2, 3), dtype=uint8, numpy=...>
```

For producing deterministic results given a `seed` value, use
<a href="../../tf/image/stateless_random_jpeg_quality.md"><code>tf.image.stateless_random_jpeg_quality</code></a>. Unlike using the `seed` param
with `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the
same results given the same seed independent of how many times the function is
called, and independent of global seed settings (e.g. tf.random.set_seed).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`image`<a id="image"></a>
</td>
<td>
3D image. Size of the last dimension must be 1 or 3.
</td>
</tr><tr>
<td>
`min_jpeg_quality`<a id="min_jpeg_quality"></a>
</td>
<td>
Minimum jpeg encoding quality to use.
</td>
</tr><tr>
<td>
`max_jpeg_quality`<a id="max_jpeg_quality"></a>
</td>
<td>
Maximum jpeg encoding quality to use.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
An operation-specific seed. It will be used in conjunction with the
graph-level seed to determine the real seeds that will be used in this
operation. Please see the documentation of set_random_seed for its
interaction with the graph-level random seed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Adjusted image(s), same shape and DType as `image`.
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
if `min_jpeg_quality` or `max_jpeg_quality` is invalid.
</td>
</tr>
</table>

