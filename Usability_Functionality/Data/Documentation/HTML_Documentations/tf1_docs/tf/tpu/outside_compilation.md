<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.outside_compilation" />
<meta itemprop="path" content="Stable" />
</div>

# tf.tpu.outside_compilation

Builds part of a computation outside any current TPU replicate scope.

### Aliases:

* `tf.compat.v1.tpu.outside_compilation`
* `tf.compat.v2.compat.v1.tpu.outside_compilation`
* `tf.contrib.tpu.outside_compilation`
* `tf.tpu.outside_compilation`

``` python
tf.tpu.outside_compilation(
    computation,
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`computation`</b>: A Python function that builds the computation to
  place on the host.
* <b>`*args`</b>: the positional arguments for the computation.
* <b>`**kwargs`</b>: the keyword arguments for the computation.


#### Returns:

The Tensors returned by computation.
