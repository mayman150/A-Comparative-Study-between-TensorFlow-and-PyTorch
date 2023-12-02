<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.div" />
<meta itemprop="path" content="Stable" />
</div>

# tf.div

Divides x / y elementwise (using Python 2 division operator semantics). (deprecated)

### Aliases:

* `tf.RaggedTensor.__div__`
* `tf.compat.v1.RaggedTensor.__div__`
* `tf.compat.v1.div`
* `tf.compat.v2.RaggedTensor.__div__`
* `tf.compat.v2.compat.v1.RaggedTensor.__div__`
* `tf.compat.v2.compat.v1.div`
* `tf.div`

``` python
tf.div(
    x,
    y,
    name=None
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.

NOTE: Prefer using the Tensor division operator or tf.divide which obey Python
3 division operator semantics.

This function divides `x` and `y`, forcing Python 2 semantics. That is, if `x`
and `y` are both integers then the result will be an integer. This is in
contrast to Python 3, where division with `/` is always a float while division
with `//` is always an integer.

#### Args:


* <b>`x`</b>: `Tensor` numerator of real numeric type.
* <b>`y`</b>: `Tensor` denominator of real numeric type.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

`x / y` returns the quotient of x and y.
