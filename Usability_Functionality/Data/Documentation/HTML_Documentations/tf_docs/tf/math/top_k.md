description: Finds values and indices of the k largest entries for the last dimension.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.top_k" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.top_k

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Finds values and indices of the `k` largest entries for the last dimension.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.nn.top_k`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.top_k`, `tf.compat.v1.nn.top_k`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.top_k(
    input,
    k=1,
    sorted=True,
    index_type=<a href="../../tf/dtypes.md#int32"><code>tf.dtypes.int32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

If the input is a vector (rank=1), finds the `k` largest entries in the vector
and outputs their values and indices as vectors.  Thus `values[j]` is the
`j`-th largest entry in `input`, and its index is `indices[j]`.

```
>>> result = tf.math.top_k([1, 2, 98, 1, 1, 99, 3, 1, 3, 96, 4, 1],
...                         k=3)
>>> result.values.numpy()
array([99, 98, 96], dtype=int32)
>>> result.indices.numpy()
array([5, 2, 9], dtype=int32)
```

For matrices (resp. higher rank input), computes the top `k` entries in each
row (resp. vector along the last dimension).  Thus,

```
>>> input = tf.random.normal(shape=(3,4,5,6))
>>> k = 2
>>> values, indices  = tf.math.top_k(input, k=k)
>>> values.shape.as_list()
[3, 4, 5, 2]
>>>
>>> values.shape == indices.shape == input.shape[:-1] + [k]
True
```

The indices can be used to `gather` from a tensor who's shape matches `input`.

```
>>> gathered_values = tf.gather(input, indices, batch_dims=-1)
>>> assert tf.reduce_all(gathered_values == values)
```

If two elements are equal, the lower-index element appears first.

```
>>> result = tf.math.top_k([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
...                        k=3)
>>> result.indices.numpy()
array([0, 1, 3], dtype=int32)
```

By default, indices are returned as type `int32`, however, this can be changed
by specifying the `index_type`.

```
>>> result = tf.math.top_k([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
...                        k=3, index_type=tf.int16)
>>> result.indices.numpy()
array([0, 1, 3], dtype=int16)
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
1-D or higher `Tensor` with last dimension at least `k`.
</td>
</tr><tr>
<td>
`k`<a id="k"></a>
</td>
<td>
0-D `Tensor` of type `int16`, `int32` or `int64`.  Number of top element
to look for along the last dimension (along each row for matrices).
</td>
</tr><tr>
<td>
`sorted`<a id="sorted"></a>
</td>
<td>
If true the resulting `k` elements will be sorted by the values in
descending order.
</td>
</tr><tr>
<td>
`index_type`<a id="index_type"></a>
</td>
<td>
Optional dtype for output indices.
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
A tuple with two named fields:
</td>
</tr>
<tr>
<td>
`values`<a id="values"></a>
</td>
<td>
The `k` largest elements along each last dimensional slice.
</td>
</tr><tr>
<td>
`indices`<a id="indices"></a>
</td>
<td>
The indices of `values` within the last dimension of `input`.
</td>
</tr>
</table>

