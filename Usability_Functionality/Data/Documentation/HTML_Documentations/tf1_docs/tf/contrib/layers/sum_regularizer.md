<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.sum_regularizer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.sum_regularizer

Returns a function that applies the sum of multiple regularizers.

``` python
tf.contrib.layers.sum_regularizer(
    regularizer_list,
    scope=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`regularizer_list`</b>: A list of regularizers to apply.
* <b>`scope`</b>: An optional scope name


#### Returns:

A function with signature `sum_reg(weights)` that applies the
sum of all the input regularizers.
