<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.merge_all" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.merge_all

Merges all summaries collected in the default graph.

### Aliases:

* `tf.compat.v1.summary.merge_all`
* `tf.compat.v2.compat.v1.summary.merge_all`
* `tf.summary.merge_all`

``` python
tf.summary.merge_all(
    key=tf.GraphKeys.SUMMARIES,
    scope=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`key`</b>: `GraphKey` used to collect the summaries.  Defaults to
  <a href="../../tf/GraphKeys.md#SUMMARIES"><code>GraphKeys.SUMMARIES</code></a>.
* <b>`scope`</b>: Optional scope used to filter the summary ops, using `re.match`


#### Returns:

If no summaries were collected, returns None.  Otherwise returns a scalar
`Tensor` of type `string` containing the serialized `Summary` protocol
buffer resulting from the merging.



#### Raises:


* <b>`RuntimeError`</b>: If called with eager execution enabled.



#### Eager Compatibility
Not compatible with eager execution. To write TensorBoard
summaries under eager execution, use <a href="../../tf/contrib/summary.md"><code>tf.contrib.summary</code></a> instead.

