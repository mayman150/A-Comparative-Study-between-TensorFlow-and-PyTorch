<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.in_cross_replica_context" />
<meta itemprop="path" content="Stable" />
</div>

# tf.distribute.in_cross_replica_context

Returns `True` if in a cross-replica context.

### Aliases:

* `tf.compat.v1.distribute.in_cross_replica_context`
* `tf.compat.v2.compat.v1.distribute.in_cross_replica_context`
* `tf.compat.v2.distribute.in_cross_replica_context`
* `tf.contrib.distribute.in_cross_replica_context`
* `tf.distribute.in_cross_replica_context`

``` python
tf.distribute.in_cross_replica_context()
```

<!-- Placeholder for "Used in" -->

See <a href="../../tf/distribute/get_replica_context.md"><code>tf.distribute.get_replica_context</code></a> for details.

```
assert not tf.distribute.in_cross_replica_context()
with strategy.scope():
  assert tf.distribute.in_cross_replica_context()

  def f():
    assert not tf.distribute.in_cross_replica_context()

  strategy.experimental_run_v2(f)
```

#### Returns:

`True` if in a cross-replica context (`get_replica_context()` returns
`None`), or `False` if in a replica context (`get_replica_context()` returns
non-`None`).
