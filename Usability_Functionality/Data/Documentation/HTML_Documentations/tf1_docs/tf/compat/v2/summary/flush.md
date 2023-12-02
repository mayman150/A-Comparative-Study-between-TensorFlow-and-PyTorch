<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v2.summary.flush" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v2.summary.flush

Forces summary writer to send any buffered data to storage.

### Aliases:

* `tf.compat.v2.summary.flush`
* `tf.contrib.summary.flush`

``` python
tf.compat.v2.summary.flush(
    writer=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->

This operation blocks until that finishes.

#### Args:


* <b>`writer`</b>: The `tf.summary.SummaryWriter` resource to flush.
  The thread default will be used if this parameter is None.
  Otherwise a <a href="../../../../tf/no_op.md"><code>tf.no_op</code></a> is returned.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created <a href="../../../../tf/Operation.md"><code>tf.Operation</code></a>.
