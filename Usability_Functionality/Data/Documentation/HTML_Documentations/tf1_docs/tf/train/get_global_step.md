<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.get_global_step" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.get_global_step

Get the global step tensor.

### Aliases:

* `tf.compat.v1.train.get_global_step`
* `tf.compat.v2.compat.v1.train.get_global_step`
* `tf.train.get_global_step`

``` python
tf.train.get_global_step(graph=None)
```

<!-- Placeholder for "Used in" -->

The global step tensor must be an integer variable. We first try to find it
in the collection `GLOBAL_STEP`, or by name `global_step:0`.

#### Args:


* <b>`graph`</b>: The graph to find the global step in. If missing, use default graph.


#### Returns:

The global step variable, or `None` if none was found.



#### Raises:


* <b>`TypeError`</b>: If the global step tensor has a non-integer type, or if it is not
  a `Variable`.