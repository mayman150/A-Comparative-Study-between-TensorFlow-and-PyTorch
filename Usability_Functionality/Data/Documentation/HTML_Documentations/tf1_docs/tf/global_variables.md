<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.global_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.global_variables

Returns global variables.

### Aliases:

* `tf.compat.v1.global_variables`
* `tf.compat.v2.compat.v1.global_variables`
* `tf.global_variables`

``` python
tf.global_variables(scope=None)
```

<!-- Placeholder for "Used in" -->

Global variables are variables that are shared across machines in a
distributed environment. The `Variable()` constructor or `get_variable()`
automatically adds new variables to the graph collection
<a href="../tf/GraphKeys.md#GLOBAL_VARIABLES"><code>GraphKeys.GLOBAL_VARIABLES</code></a>.
This convenience function returns the contents of that collection.

An alternative to global variables are local variables. See
<a href="../tf/local_variables.md"><code>tf.compat.v1.local_variables</code></a>

#### Args:


* <b>`scope`</b>: (Optional.) A string. If supplied, the resulting list is filtered to
  include only items whose `name` attribute matches `scope` using
  `re.match`. Items without a `name` attribute are never returned if a scope
  is supplied. The choice of `re.match` means that a `scope` without special
  tokens filters by prefix.


#### Returns:

A list of `Variable` objects.
