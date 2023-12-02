<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.main_op.main_op" />
<meta itemprop="path" content="Stable" />
</div>

# tf.saved_model.main_op.main_op

Returns a main op to init variables and tables. (deprecated)

### Aliases:

* `tf.compat.v1.saved_model.main_op.main_op`
* `tf.compat.v2.compat.v1.saved_model.main_op.main_op`
* `tf.saved_model.main_op.main_op`

``` python
tf.saved_model.main_op.main_op()
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.main_op.main_op.

Returns the main op including the group of ops that initializes all
variables, initializes local variables and initialize all tables.

#### Returns:

The set of ops to be run as part of the main op upon the load operation.
