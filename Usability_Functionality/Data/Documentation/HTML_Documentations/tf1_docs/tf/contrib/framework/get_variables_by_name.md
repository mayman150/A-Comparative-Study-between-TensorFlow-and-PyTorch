<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.get_variables_by_name" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.get_variables_by_name

Gets the list of variables that were given that name.

``` python
tf.contrib.framework.get_variables_by_name(
    given_name,
    scope=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`given_name`</b>: name given to the variable without any scope.
* <b>`scope`</b>: an optional scope for filtering the variables to return.


#### Returns:

a copied list of variables with the given name and scope.
