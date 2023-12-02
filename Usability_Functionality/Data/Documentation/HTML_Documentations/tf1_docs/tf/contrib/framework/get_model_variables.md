<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.get_model_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.get_model_variables

Gets the list of model variables, filtered by scope and/or suffix.

``` python
tf.contrib.framework.get_model_variables(
    scope=None,
    suffix=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`scope`</b>: an optional scope for filtering the variables to return.
* <b>`suffix`</b>: an optional suffix for filtering the variables to return.


#### Returns:

a list of variables in collection with scope and suffix.
