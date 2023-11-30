description: Removes dimensions of size 1 from the shape of a tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.squeeze" />
<meta itemprop="path" content="Stable" />
</div>

# tf.squeeze

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Removes dimensions of size 1 from the shape of a tensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.squeeze(
    input, axis=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given a tensor `input`, this operation returns a tensor of the same type with
all dimensions of size 1 removed. If you don't want to remove all size 1
dimensions, you can remove specific size 1 dimensions by specifying
`axis`.

#### For example:



```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t))  # [2, 3]
```

Or, to remove specific size 1 dimensions:

```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
```

Unlike the older op <a href="../tf/compat/v1/squeeze.md"><code>tf.compat.v1.squeeze</code></a>, this op does not accept a
deprecated `squeeze_dims` argument.

Note: if `input` is a <a href="../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>, then this operation takes `O(N)`
time, where `N` is the number of elements in the squeezed dimensions.

Note: If squeeze is performed on dimensions of unknown sizes, then the
returned Tensor will be of unknown shape. A common situation is when the
first (batch) dimension is of size `None`, <a href="../tf/squeeze.md"><code>tf.squeeze</code></a> returns
`<unknown>` shape which may be a surprise. Specify the `axis=` argument
to get the expected result, as illustrated in the following example:

```python
@tf.function
def func(x):
  print('x.shape:', x.shape)
  known_axes = [i for i, size in enumerate(x.shape) if size == 1]
  y = tf.squeeze(x, axis=known_axes)
  print('shape of tf.squeeze(x, axis=known_axes):', y.shape)
  y = tf.squeeze(x)
  print('shape of tf.squeeze(x):', y.shape)
  return 0

_ = func.get_concrete_function(tf.TensorSpec([None, 1, 2], dtype=tf.int32))
# Output is.
# x.shape: (None, 1, 2)
# shape of tf.squeeze(x, axis=known_axes): (None, 2)
# shape of tf.squeeze(x): <unknown>
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
A `Tensor`. The `input` to squeeze.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
An optional list of `ints`. Defaults to `[]`. If specified, only
squeezes the dimensions listed. The dimension index starts at 0. It is an
error to squeeze a dimension that is not 1. Must be in the range
`[-rank(input), rank(input))`. Must be specified if `input` is a
`RaggedTensor`.
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
A `Tensor`. Has the same type as `input`.
Contains the same data as `input`, but has one or more dimensions of
size 1 removed.
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
The input cannot be converted to a tensor, or the specified
axis cannot be squeezed.
</td>
</tr>
</table>

