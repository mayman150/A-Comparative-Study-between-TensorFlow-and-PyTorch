description: Transposes a SparseTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sparse.transpose" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sparse.transpose

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/sparse_ops.py">View source</a>



Transposes a `SparseTensor`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sparse.transpose`, `tf.compat.v1.sparse_transpose`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sparse.transpose(
    sp_input, perm=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Permutes the dimensions according to the value of `perm`.  This is the sparse
version of <a href="../../tf/transpose.md"><code>tf.transpose</code></a>.

The returned tensor's dimension `i` will correspond to the input dimension
`perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is the rank
of the input tensor. Hence, by default, this operation performs a regular
matrix transpose on 2-D input Tensors.

#### For example:



```
>>> x = tf.SparseTensor(indices=[[0, 1], [0, 3], [2, 3], [3, 1]],
...                     values=[1.1, 2.2, 3.3, 4.4],
...                     dense_shape=[4, 5])
>>> print('x =', tf.sparse.to_dense(x))
x = tf.Tensor(
[[0.  1.1 0.  2.2 0. ]
[0.  0.  0.  0.  0. ]
[0.  0.  0.  3.3 0. ]
[0.  4.4 0.  0.  0. ]], shape=(4, 5), dtype=float32)
```

```
>>> x_transpose = tf.sparse.transpose(x)
>>> print('x_transpose =', tf.sparse.to_dense(x_transpose))
x_transpose = tf.Tensor(
[[0.  0.  0.  0. ]
[1.1 0.  0.  4.4]
[0.  0.  0.  0. ]
[2.2 0.  3.3 0. ]
[0.  0.  0.  0. ]], shape=(5, 4), dtype=float32)
```

Equivalently, you could call `tf.sparse.transpose(x, perm=[1, 0])`.  The
`perm` argument is more useful for n-dimensional tensors where n > 2.

```
>>> x = tf.SparseTensor(indices=[[0, 0, 1], [0, 0, 3], [1, 2, 3], [1, 3, 1]],
...                     values=[1.1, 2.2, 3.3, 4.4],
...                     dense_shape=[2, 4, 5])
>>> print('x =', tf.sparse.to_dense(x))
x = tf.Tensor(
[[[0.  1.1 0.  2.2 0. ]
  [0.  0.  0.  0.  0. ]
  [0.  0.  0.  0.  0. ]
  [0.  0.  0.  0.  0. ]]
[[0.  0.  0.  0.  0. ]
  [0.  0.  0.  0.  0. ]
  [0.  0.  0.  3.3 0. ]
  [0.  4.4 0.  0.  0. ]]], shape=(2, 4, 5), dtype=float32)
```

As above, simply calling <a href="../../tf/sparse/transpose.md"><code>tf.sparse.transpose</code></a> will default to `perm=[2,1,0]`.

To take the transpose of a batch of sparse matrices, where 0 is the batch
dimension, you would set `perm=[0,2,1]`.

```
>>> x_transpose = tf.sparse.transpose(x, perm=[0, 2, 1])
>>> print('x_transpose =', tf.sparse.to_dense(x_transpose))
x_transpose = tf.Tensor(
[[[0.  0.  0.  0. ]
  [1.1 0.  0.  0. ]
  [0.  0.  0.  0. ]
  [2.2 0.  0.  0. ]
  [0.  0.  0.  0. ]]
[[0.  0.  0.  0. ]
  [0.  0.  0.  4.4]
  [0.  0.  0.  0. ]
  [0.  0.  3.3 0. ]
  [0.  0.  0.  0. ]]], shape=(2, 5, 4), dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sp_input`<a id="sp_input"></a>
</td>
<td>
The input `SparseTensor`.
</td>
</tr><tr>
<td>
`perm`<a id="perm"></a>
</td>
<td>
A permutation vector of the dimensions of `sp_input`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name prefix for the returned tensors (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A transposed `SparseTensor`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
If `sp_input` is not a `SparseTensor`.
</td>
</tr>
</table>

