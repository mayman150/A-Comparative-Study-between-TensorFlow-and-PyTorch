<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.initializers.local_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.initializers.local_variables

Returns an Op that initializes all local variables.

### Aliases:

* `tf.compat.v1.initializers.local_variables`
* `tf.compat.v1.local_variables_initializer`
* `tf.compat.v2.compat.v1.initializers.local_variables`
* `tf.compat.v2.compat.v1.local_variables_initializer`
* `tf.initializers.local_variables`
* `tf.local_variables_initializer`

``` python
tf.initializers.local_variables()
```

<!-- Placeholder for "Used in" -->

This is just a shortcut for `variables_initializer(local_variables())`

#### Returns:

An Op that initializes all local variables in the graph.
