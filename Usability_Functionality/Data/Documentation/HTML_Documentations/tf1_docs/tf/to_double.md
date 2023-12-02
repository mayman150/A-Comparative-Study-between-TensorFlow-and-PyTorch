<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.to_double" />
<meta itemprop="path" content="Stable" />
</div>

# tf.to_double

Casts a tensor to type `float64`. (deprecated)

### Aliases:

* `tf.compat.v1.to_double`
* `tf.compat.v2.compat.v1.to_double`
* `tf.to_double`

``` python
tf.to_double(
    x,
    name='ToDouble'
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
type `float64`.



#### Raises:


* <b>`TypeError`</b>: If `x` cannot be cast to the `float64`.