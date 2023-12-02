<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.core" />
<meta itemprop="path" content="Stable" />
</div>

# tf.tpu.core

Returns the device name for a core in a replicated TPU computation.

### Aliases:

* `tf.compat.v1.tpu.core`
* `tf.compat.v2.compat.v1.tpu.core`
* `tf.contrib.tpu.core`
* `tf.tpu.core`

``` python
tf.tpu.core(num)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`num`</b>: the virtual core number within each replica to which operators should
be assigned.

#### Returns:

A device name, suitable for passing to <a href="../../tf/device.md"><code>tf.device()</code></a>.
