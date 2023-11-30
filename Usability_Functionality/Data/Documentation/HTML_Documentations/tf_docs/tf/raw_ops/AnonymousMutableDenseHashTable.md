description: Creates an empty anonymous mutable hash table that uses tensors as the backing store.
robots: noindex

# tf.raw_ops.AnonymousMutableDenseHashTable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates an empty anonymous mutable hash table that uses tensors as the backing store.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.AnonymousMutableDenseHashTable`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.AnonymousMutableDenseHashTable(
    empty_key,
    deleted_key,
    value_dtype,
    value_shape=[],
    initial_num_buckets=131072,
    max_load_factor=0.8,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This op creates a new anonymous mutable hash table (as a resource) everytime
it is executed, with the specified dtype of its keys and values,
returning the resource handle. Each value must be a scalar.
Data can be inserted into the table using
the insert operations. It does not support the initialization operation.

It uses "open addressing" with quadratic reprobing to resolve
collisions.

The table is anonymous in the sense that it can only be
accessed by the returned resource handle (e.g. it cannot be looked up
by a name in a resource manager). The table will be automatically
deleted when all resource handles pointing to it are gone.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`empty_key`<a id="empty_key"></a>
</td>
<td>
A `Tensor`.
The key used to represent empty key buckets internally. Must not
be used in insert or lookup operations.
</td>
</tr><tr>
<td>
`deleted_key`<a id="deleted_key"></a>
</td>
<td>
A `Tensor`. Must have the same type as `empty_key`.
</td>
</tr><tr>
<td>
`value_dtype`<a id="value_dtype"></a>
</td>
<td>
A <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a>. Type of the table values.
</td>
</tr><tr>
<td>
`value_shape`<a id="value_shape"></a>
</td>
<td>
An optional <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`. Defaults to `[]`.
The shape of each value.
</td>
</tr><tr>
<td>
`initial_num_buckets`<a id="initial_num_buckets"></a>
</td>
<td>
An optional `int`. Defaults to `131072`.
The initial number of hash table buckets. Must be a power
to 2.
</td>
</tr><tr>
<td>
`max_load_factor`<a id="max_load_factor"></a>
</td>
<td>
An optional `float`. Defaults to `0.8`.
The maximum ratio between number of entries and number of
buckets before growing the table. Must be between 0 and 1.
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
A `Tensor` of type `resource`.
</td>
</tr>

</table>

