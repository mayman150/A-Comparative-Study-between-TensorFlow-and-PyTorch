<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.get_default_graph" />
<meta itemprop="path" content="Stable" />
</div>

# tf.get_default_graph

Returns the default graph for the current thread.

### Aliases:

* `tf.compat.v1.get_default_graph`
* `tf.compat.v2.compat.v1.get_default_graph`
* `tf.get_default_graph`

``` python
tf.get_default_graph()
```

<!-- Placeholder for "Used in" -->

The returned graph will be the innermost graph on which a
<a href="../tf/Graph.md#as_default"><code>Graph.as_default()</code></a> context has been entered, or a global default
graph if none has been explicitly created.

NOTE: The default graph is a property of the current thread. If you
create a new thread, and wish to use the default graph in that
thread, you must explicitly add a `with g.as_default():` in that
thread's function.

#### Returns:

The default `Graph` being used in the current thread.
