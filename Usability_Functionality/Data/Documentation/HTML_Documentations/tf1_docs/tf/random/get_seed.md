<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.get_seed" />
<meta itemprop="path" content="Stable" />
</div>

# tf.random.get_seed

Returns the local seeds an operation should use given an op-specific seed.

### Aliases:

* `tf.compat.v1.get_seed`
* `tf.compat.v1.random.get_seed`
* `tf.compat.v2.compat.v1.get_seed`
* `tf.compat.v2.compat.v1.random.get_seed`
* `tf.get_seed`
* `tf.random.get_seed`

``` python
tf.random.get_seed(op_seed)
```

<!-- Placeholder for "Used in" -->

Given operation-specific seed, `op_seed`, this helper function returns two
seeds derived from graph-level and op-level seeds. Many random operations
internally use the two seeds to allow user to change the seed globally for a
graph, or for only specific operations.

For details on how the graph-level seed interacts with op seeds, see
<a href="../../tf/random/set_random_seed.md"><code>tf.compat.v1.random.set_random_seed</code></a>.

#### Args:


* <b>`op_seed`</b>: integer.


#### Returns:

A tuple of two integers that should be used for the local seed of this
operation.
