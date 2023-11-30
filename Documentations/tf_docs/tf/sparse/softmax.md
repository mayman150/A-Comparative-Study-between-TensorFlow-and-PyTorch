description: Applies softmax to a batched N-D SparseTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sparse.softmax" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sparse.softmax

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/sparse_ops.py">View source</a>



Applies softmax to a batched N-D `SparseTensor`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sparse.softmax`, `tf.compat.v1.sparse_softmax`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sparse.softmax(
    sp_input, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The inputs represent an N-D SparseTensor with logical shape `[..., B, C]`
(where `N >= 2`), and with indices sorted in the canonical lexicographic
order.

This op is equivalent to applying the normal <a href="../../tf/nn/softmax.md"><code>tf.nn.softmax()</code></a> to each
innermost logical submatrix with shape `[B, C]`, but with the catch that *the
implicitly zero elements do not participate*.  Specifically, the algorithm is
equivalent to:

  (1) Applies <a href="../../tf/nn/softmax.md"><code>tf.nn.softmax()</code></a> to a densified view of each innermost
      submatrix with shape `[B, C]`, along the size-C dimension;
  (2) Masks out the original implicitly-zero locations;
  (3) Renormalizes the remaining elements.

Hence, the `SparseTensor` result has exactly the same non-zero indices and
shape.

Example using a 3-D SparseTensor:

  ```
  >>> st = tf.sparse.from_dense(
  ...   [[[0., np.e],
  ...     [1., 0.]],
  ...
  ...    [[np.e, 0.],
  ...     [np.e, np.e]]])
  >>> res = tf.sparse.softmax(st)
  >>> res.indices
  <tf.Tensor: shape=(5, 3), dtype=int64, numpy=
  array([[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]])>
  >>> res.values
  <tf.Tensor: ... numpy=array([1. , 1. , 1. , 0.5, 0.5], dtype=float32)>
  >>> res.dense_shape
  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 2, 2])>
  >>> tf.sparse.to_dense(res)
  <tf.Tensor: shape=(2, 2, 2), dtype=float32, numpy=
  array([[[0. , 1. ],
          [1. , 0. ]],
         [[1. , 0. ],
          [0.5, 0.5]]], dtype=float32)>
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
N-D `SparseTensor`, where `N >= 2`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
optional name of the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`output`<a id="output"></a>
</td>
<td>
N-D `SparseTensor` representing the results.
</td>
</tr>
</table>

