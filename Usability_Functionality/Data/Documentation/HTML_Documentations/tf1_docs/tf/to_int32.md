<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.to_int32" />
<meta itemprop="path" content="Stable" />
</div>

# tf.to_int32

Casts a tensor to type `int32`. (deprecated)

### Aliases:

* `tf.compat.v1.to_int32`
* `tf.compat.v2.compat.v1.to_int32`
* `tf.to_int32`

``` python
tf.to_int32(
    x,
    name='ToInt32'
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
type `int32`.



#### Raises:


* <b>`TypeError`</b>: If `x` cannot be cast to the `int32`.