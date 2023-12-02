<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.eager.get_optimizer_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.eager.get_optimizer_variables

Returns a list of variables for the given <a href="../../../tf/train/Optimizer.md"><code>tf.compat.v1.train.Optimizer</code></a>.

``` python
tf.contrib.eager.get_optimizer_variables(optimizer)
```

<!-- Placeholder for "Used in" -->

Equivalent to `optimizer.variables()`.

#### Args:


* <b>`optimizer`</b>: An instance of <a href="../../../tf/train/Optimizer.md"><code>tf.compat.v1.train.Optimizer</code></a> which has created
  variables (typically after a call to `Optimizer.minimize`).


#### Returns:

A list of variables which have been created by the `Optimizer`.
