<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.distribute.get_cross_replica_context" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.distribute.get_cross_replica_context

Returns the current tf.distribute.Strategy if in a cross-replica context.

``` python
tf.contrib.distribute.get_cross_replica_context()
```

<!-- Placeholder for "Used in" -->

DEPRECATED: Please use `in_cross_replica_context()` and
`get_strategy()` instead.

#### Returns:

Returns the current <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> object in a cross-replica
context, or `None`.

Exactly one of `get_replica_context()` and `get_cross_replica_context()`
will return `None` in a particular block.
