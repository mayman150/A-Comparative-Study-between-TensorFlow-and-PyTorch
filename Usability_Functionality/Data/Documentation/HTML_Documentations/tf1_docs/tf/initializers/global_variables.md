<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.initializers.global_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.initializers.global_variables

Returns an Op that initializes global variables.

### Aliases:

* `tf.compat.v1.global_variables_initializer`
* `tf.compat.v1.initializers.global_variables`
* `tf.compat.v2.compat.v1.global_variables_initializer`
* `tf.compat.v2.compat.v1.initializers.global_variables`
* `tf.global_variables_initializer`
* `tf.initializers.global_variables`

``` python
tf.initializers.global_variables()
```

<!-- Placeholder for "Used in" -->

This is just a shortcut for `variables_initializer(global_variables())`

#### Returns:

An Op that initializes global variables in the graph.
