<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.initializers.variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.initializers.variables

Returns an Op that initializes a list of variables.

### Aliases:

* `tf.compat.v1.initializers.variables`
* `tf.compat.v1.variables_initializer`
* `tf.compat.v2.compat.v1.initializers.variables`
* `tf.compat.v2.compat.v1.variables_initializer`
* `tf.initializers.variables`
* `tf.variables_initializer`

``` python
tf.initializers.variables(
    var_list,
    name='init'
)
```

<!-- Placeholder for "Used in" -->

After you launch the graph in a session, you can run the returned Op to
initialize all the variables in `var_list`. This Op runs all the
initializers of the variables in `var_list` in parallel.

Calling `initialize_variables()` is equivalent to passing the list of
initializers to `Group()`.

If `var_list` is empty, however, the function still returns an Op that can
be run. That Op just has no effect.

#### Args:


* <b>`var_list`</b>: List of `Variable` objects to initialize.
* <b>`name`</b>: Optional name for the returned operation.


#### Returns:

An Op that run the initializers of all the specified variables.
