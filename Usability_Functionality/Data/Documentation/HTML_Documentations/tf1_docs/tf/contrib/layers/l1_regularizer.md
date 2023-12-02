<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.l1_regularizer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.l1_regularizer

Returns a function that can be used to apply L1 regularization to weights.

``` python
tf.contrib.layers.l1_regularizer(
    scale,
    scope=None
)
```

<!-- Placeholder for "Used in" -->

L1 regularization encourages sparsity.

#### Args:


* <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
* <b>`scope`</b>: An optional scope name.


#### Returns:

A function with signature `l1(weights)` that apply L1 regularization.



#### Raises:


* <b>`ValueError`</b>: If scale is negative or if scale is not a float.