<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.device" />
<meta itemprop="path" content="Stable" />
</div>

# tf.device

Wrapper for <a href="../tf/Graph.md#device"><code>Graph.device()</code></a> using the default graph.

### Aliases:

* `tf.compat.v1.device`
* `tf.compat.v2.compat.v1.device`
* `tf.device`

``` python
tf.device(device_name_or_function)
```

<!-- Placeholder for "Used in" -->

See <a href="../tf/Graph.md#device"><code>tf.Graph.device</code></a> for more details.

#### Args:


* <b>`device_name_or_function`</b>: The device name or function to use in the context.


#### Returns:

A context manager that specifies the default device to use for newly
created ops.



#### Raises:


* <b>`RuntimeError`</b>: If eager execution is enabled and a function is passed in.