description: Broadcast an array for a compatible shape.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.broadcast_to" />
<meta itemprop="path" content="Stable" />
</div>

# tf.broadcast_to

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Broadcast an array for a compatible shape.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.broadcast_to`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.broadcast_to(
    input: Annotated[Any, TV_BroadcastTo_T],
    shape: Annotated[Any, TV_BroadcastTo_Tidx],
    name=None
) -> Annotated[Any, TV_BroadcastTo_T]
</code></pre>



<!-- Placeholder for "Used in" -->

Broadcasting is the process of making arrays to have compatible shapes
for arithmetic operations. Two shapes are compatible if for each
dimension pair they are either equal or one of them is one.

#### For example:



```
>>> x = tf.constant([[1, 2, 3]])   # Shape (1, 3,)
>>> y = tf.broadcast_to(x, [2, 3])
>>> print(y)
tf.Tensor(
    [[1 2 3]
     [1 2 3]], shape=(2, 3), dtype=int32)
```

In the above example, the input Tensor with the shape of `[1, 3]`
is broadcasted to output Tensor with shape of `[2, 3]`.

When broadcasting, if a tensor has fewer axes than necessary its shape is
padded on the left with ones. So this gives the same result as the previous
example:

```
>>> x = tf.constant([1, 2, 3])   # Shape (3,)
>>> y = tf.broadcast_to(x, [2, 3])
```


When doing broadcasted operations such as multiplying a tensor
by a scalar, broadcasting (usually) confers some time or space
benefit, as the broadcasted tensor is never materialized.

However, `broadcast_to` does not carry with it any such benefits.
The newly-created tensor takes the full memory of the broadcasted
shape. (In a graph context, `broadcast_to` might be fused to
subsequent operation and then be optimized away, however.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. A Tensor to broadcast.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
An 1-D `int` Tensor. The shape of the desired output.
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
</td>
</tr>

</table>

