<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.get_unique_variable" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.get_unique_variable

Gets the variable uniquely identified by that var_op_name.

``` python
tf.contrib.framework.get_unique_variable(var_op_name)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`var_op_name`</b>: the full name of the variable op, including the scope.


#### Returns:

a tensorflow variable.



#### Raises:


* <b>`ValueError`</b>: if no variable uniquely identified by the name exists.