description: Returns the diagonal part of the tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.linalg.tensor_diag_part" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.linalg.tensor_diag_part

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Returns the diagonal part of the tensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.linalg.tensor_diag_part(
    input, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation returns a tensor with the `diagonal` part
of the `input`. The `diagonal` part is computed as follows:

Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
tensor of rank `k` with dimensions `[D1,..., Dk]` where:

`diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

For a rank 2 tensor, `linalg.diag_part` and `linalg.tensor_diag_part`
produce the same result. For rank 3 and higher, linalg.diag_part extracts
the diagonal of each inner-most matrix in the tensor. An example where
they differ is given below.

```
>>> x = [[[[1111,1112],[1121,1122]],
...       [[1211,1212],[1221,1222]]],
...      [[[2111, 2112], [2121, 2122]],
...       [[2211, 2212], [2221, 2222]]]
...      ]
>>> tf.linalg.tensor_diag_part(x)
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1111, 1212],
       [2121, 2222]], dtype=int32)>
>>> tf.linalg.diag_part(x).shape
TensorShape([2, 2, 2])
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
A `Tensor` with rank `2k`.
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
A Tensor containing diagonals of `input`. Has the same type as `input`, and
rank `k`.
</td>
</tr>

</table>

