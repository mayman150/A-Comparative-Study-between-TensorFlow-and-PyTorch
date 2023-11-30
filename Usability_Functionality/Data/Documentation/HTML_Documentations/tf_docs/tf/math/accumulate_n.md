description: Returns the element-wise sum of a list of tensors. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.accumulate_n" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.accumulate_n

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Returns the element-wise sum of a list of tensors. (deprecated)


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.accumulate_n`, `tf.compat.v1.math.accumulate_n`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.accumulate_n(
    inputs, shape=None, tensor_dtype=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../tf/math/add_n.md"><code>tf.math.add_n</code></a> Instead

Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
otherwise, these are inferred.

#### For example:



```
>>> a = tf.constant([[1, 2], [3, 4]])
>>> b = tf.constant([[5, 0], [0, 6]])
>>> tf.math.accumulate_n([a, b, a]).numpy()
array([[ 7, 4],
       [ 6, 14]], dtype=int32)
```

```
>>> # Explicitly pass shape and type
>>> tf.math.accumulate_n(
...     [a, b, a], shape=[2, 2], tensor_dtype=tf.int32).numpy()
array([[ 7,  4],
       [ 6, 14]], dtype=int32)
```

Note: The input must be a list or tuple. This function does not handle
`IndexedSlices`

#### See Also:



* <a href="../../tf/math/reduce_sum.md"><code>tf.reduce_sum(inputs, axis=0)</code></a> - This performe the same mathematical
  operation, but <a href="../../tf/math/add_n.md"><code>tf.add_n</code></a> may be more efficient because it sums the
  tensors directly. `reduce_sum` on the other hand calls
  <a href="../../tf/convert_to_tensor.md"><code>tf.convert_to_tensor</code></a> on the list of tensors, unncessairly stacking them
  into a single tensor before summing.
* <a href="../../tf/math/add_n.md"><code>tf.add_n</code></a> - This is another python wrapper for the same Op. It has
  nearly identical functionality.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A list of `Tensor` objects, each with same shape and type.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
Expected shape of elements of `inputs` (optional). Also controls the
output shape of this op, which may affect type inference in other ops. A
value of `None` means "infer the input shape from the shapes in `inputs`".
</td>
</tr><tr>
<td>
`tensor_dtype`<a id="tensor_dtype"></a>
</td>
<td>
Expected data type of `inputs` (optional). A value of `None`
means "infer the input dtype from `inputs[0]`".
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
A `Tensor` of same shape and type as the elements of `inputs`.
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
If `inputs` don't all have same shape and dtype or the shape
cannot be inferred.
</td>
</tr>
</table>

