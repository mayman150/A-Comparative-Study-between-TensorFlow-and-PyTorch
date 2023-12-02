<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.smart_constant_value" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.smart_constant_value

Return the bool value for `pred`, or None if `pred` had a dynamic value.

``` python
tf.contrib.framework.smart_constant_value(pred)
```

<!-- Placeholder for "Used in" -->


#### Arguments:


* <b>`pred`</b>: A scalar, either a Python bool or tensor.


#### Returns:

True or False if `pred` has a constant boolean value, None otherwise.



#### Raises:


* <b>`TypeError`</b>: If `pred` is not a Tensor or bool.