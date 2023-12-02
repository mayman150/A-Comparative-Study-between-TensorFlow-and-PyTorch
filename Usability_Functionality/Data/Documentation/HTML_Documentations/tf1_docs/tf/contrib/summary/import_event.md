<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.summary.import_event" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.summary.import_event

Writes a <a href="../../../tf/Event.md"><code>tf.compat.v1.Event</code></a> binary proto.

``` python
tf.contrib.summary.import_event(
    tensor,
    name=None
)
```

<!-- Placeholder for "Used in" -->

This can be used to import existing event logs into a new summary writer sink.
Please note that this is lower level than the other summary functions and
will ignore the `tf.summary.should_record_summaries` setting.

#### Args:


* <b>`tensor`</b>: A <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> of type `string` containing a serialized
  <a href="../../../tf/Event.md"><code>tf.compat.v1.Event</code></a> proto.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created <a href="../../../tf/Operation.md"><code>tf.Operation</code></a>.
