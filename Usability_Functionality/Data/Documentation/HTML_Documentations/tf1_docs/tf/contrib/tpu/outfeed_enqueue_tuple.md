<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.tpu.outfeed_enqueue_tuple" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.tpu.outfeed_enqueue_tuple

Enqueue multiple Tensor values on the computation outfeed.

``` python
tf.contrib.tpu.outfeed_enqueue_tuple(
    inputs,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`inputs`</b>: A list of `Tensor` objects.
  A list of tensors that will be inserted into the outfeed queue as an
  XLA tuple.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.
