description: Returns shape of a list of tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.shape_n" />
<meta itemprop="path" content="Stable" />
</div>

# tf.shape_n

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Returns shape of a list of tensors.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.shape_n`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.shape_n(
    input,
    out_type=<a href="../tf/dtypes.md#int32"><code>tf.dtypes.int32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a list of tensors, <a href="../tf/shape_n.md"><code>tf.shape_n</code></a> is much faster than applying <a href="../tf/shape.md"><code>tf.shape</code></a>
to each tensor individually.
```
>>> a = tf.ones([1, 2])
>>> b = tf.ones([2, 3])
>>> c = tf.ones([3, 4])
>>> tf.shape_n([a, b, c])
[<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>,
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 3], dtype=int32)>,
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A list of at least 1 `Tensor` object with the same dtype.
</td>
</tr><tr>
<td>
`out_type`<a id="out_type"></a>
</td>
<td>
The specified output type of the operation (`int32` or `int64`).
Defaults to <a href="../tf.md#int32"><code>tf.int32</code></a>(optional).
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of `Tensor` specifying the shape of each input tensor with type of
`out_type`.
</td>
</tr>

</table>

