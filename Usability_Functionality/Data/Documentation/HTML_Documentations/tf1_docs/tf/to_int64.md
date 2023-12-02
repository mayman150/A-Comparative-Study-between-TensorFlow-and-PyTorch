<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.to_int64" />
<meta itemprop="path" content="Stable" />
</div>

# tf.to_int64

Casts a tensor to type `int64`. (deprecated)

### Aliases:

* `tf.compat.v1.to_int64`
* `tf.compat.v2.compat.v1.to_int64`
* `tf.to_int64`

``` python
tf.to_int64(
    x,
    name='ToInt64'
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../tf/cast.md"><code>tf.cast</code></a> instead.

#### Args:


* <b>`x`</b>: A `Tensor` or `SparseTensor` or `IndexedSlices`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` with
type `int64`.



#### Raises:


* <b>`TypeError`</b>: If `x` cannot be cast to the `int64`.