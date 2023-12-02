<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.shutdown_system" />
<meta itemprop="path" content="Stable" />
</div>

# tf.tpu.shutdown_system

Shuts down a running a distributed TPU system.

### Aliases:

* `tf.compat.v1.tpu.shutdown_system`
* `tf.compat.v2.compat.v1.tpu.shutdown_system`
* `tf.contrib.tpu.shutdown_system`
* `tf.tpu.shutdown_system`

``` python
tf.tpu.shutdown_system(job=None)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`job`</b>: The job (the XXX in TensorFlow device specification /job:XXX) that
  contains the TPU devices that will be shutdown. If job=None it is
  assumed there is only one job in the TensorFlow flock, and an error will
  be returned if this assumption does not hold.