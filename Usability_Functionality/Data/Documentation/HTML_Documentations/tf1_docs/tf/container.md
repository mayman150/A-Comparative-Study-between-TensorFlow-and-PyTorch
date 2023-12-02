<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.container" />
<meta itemprop="path" content="Stable" />
</div>

# tf.container

Wrapper for <a href="../tf/Graph.md#container"><code>Graph.container()</code></a> using the default graph.

### Aliases:

* `tf.compat.v1.container`
* `tf.compat.v2.compat.v1.container`
* `tf.container`

``` python
tf.container(container_name)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`container_name`</b>: The container string to use in the context.


#### Returns:

A context manager that specifies the default container to use for newly
created stateful ops.
