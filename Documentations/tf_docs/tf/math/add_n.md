description: Returns the element-wise sum of a list of tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.add_n" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.add_n

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Returns the element-wise sum of a list of tensors.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.add_n`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.add_n`, `tf.compat.v1.math.add_n`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.add_n(
    inputs, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

All inputs in the list must have the same shape. This op does not
[broadcast](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
its inputs. If you need broadcasting, use <a href="../../tf/math/add.md"><code>tf.math.add</code></a> (or the `+` operator)
instead.

#### For example:



```
>>> a = tf.constant([[3, 5], [4, 8]])
>>> b = tf.constant([[1, 6], [2, 9]])
>>> tf.math.add_n([a, b, a]).numpy()
array([[ 7, 16],
       [10, 25]], dtype=int32)
```

#### See Also:



* <a href="../../tf/math/reduce_sum.md"><code>tf.reduce_sum(inputs, axis=0)</code></a> - This performs the same mathematical
  operation, but <a href="../../tf/math/add_n.md"><code>tf.add_n</code></a> may be more efficient because it sums the
  tensors directly. `reduce_sum` on the other hand calls
  <a href="../../tf/convert_to_tensor.md"><code>tf.convert_to_tensor</code></a> on the list of tensors, unnecessarily stacking them
  into a single tensor before summing.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A list of <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> or <a href="../../tf/IndexedSlices.md"><code>tf.IndexedSlices</code></a> objects, each with the
same shape and type. <a href="../../tf/IndexedSlices.md"><code>tf.IndexedSlices</code></a> objects will be converted into
dense tensors prior to adding.
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
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> of the same shape and type as the elements of `inputs`.
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

