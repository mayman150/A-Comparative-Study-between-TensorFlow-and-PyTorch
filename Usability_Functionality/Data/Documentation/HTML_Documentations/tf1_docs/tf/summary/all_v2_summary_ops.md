<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.all_v2_summary_ops" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.all_v2_summary_ops

Returns all V2-style summary ops defined in the current default graph.

### Aliases:

* `tf.compat.v1.summary.all_v2_summary_ops`
* `tf.compat.v2.compat.v1.summary.all_v2_summary_ops`
* `tf.contrib.summary.all_summary_ops`
* `tf.summary.all_v2_summary_ops`

``` python
tf.summary.all_v2_summary_ops()
```

<!-- Placeholder for "Used in" -->

This includes ops from TF 2.0 tf.summary and TF 1.x tf.contrib.summary (except
for <a href="../../tf/contrib/summary/graph.md"><code>tf.contrib.summary.graph</code></a> and <a href="../../tf/contrib/summary/import_event.md"><code>tf.contrib.summary.import_event</code></a>), but
does *not* include TF 1.x tf.summary ops.

#### Returns:

List of summary ops, or None if called under eager execution.
