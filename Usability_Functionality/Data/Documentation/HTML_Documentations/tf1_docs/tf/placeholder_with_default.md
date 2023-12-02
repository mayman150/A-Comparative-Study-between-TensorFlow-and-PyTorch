<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.placeholder_with_default" />
<meta itemprop="path" content="Stable" />
</div>

# tf.placeholder_with_default

A placeholder op that passes through `input` when its output is not fed.

### Aliases:

* `tf.compat.v1.placeholder_with_default`
* `tf.compat.v2.compat.v1.placeholder_with_default`
* `tf.placeholder_with_default`

``` python
tf.placeholder_with_default(
    input,
    shape,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`input`</b>: A `Tensor`. The default value to produce when output is not fed.
* <b>`shape`</b>: A <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `int`s. The (possibly partial) shape of
  the tensor.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor`. Has the same type as `input`.
