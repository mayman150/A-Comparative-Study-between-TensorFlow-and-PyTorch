<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.add_to_collection" />
<meta itemprop="path" content="Stable" />
</div>

# tf.add_to_collection

Wrapper for <a href="../tf/Graph.md#add_to_collection"><code>Graph.add_to_collection()</code></a> using the default graph.

### Aliases:

* `tf.add_to_collection`
* `tf.compat.v1.add_to_collection`
* `tf.compat.v2.compat.v1.add_to_collection`

``` python
tf.add_to_collection(
    name,
    value
)
```

<!-- Placeholder for "Used in" -->

See <a href="../tf/Graph.md#add_to_collection"><code>tf.Graph.add_to_collection</code></a>
for more details.

#### Args:


* <b>`name`</b>: The key for the collection. For example, the `GraphKeys` class
  contains many standard names for collections.
* <b>`value`</b>: The value to add to the collection.

#### Eager Compatibility
Collections are only supported in eager when variables are created inside
an EagerVariableStore (e.g. as part of a layer or template).

