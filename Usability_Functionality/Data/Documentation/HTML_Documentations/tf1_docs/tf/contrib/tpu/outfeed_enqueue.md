<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.tpu.outfeed_enqueue" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.tpu.outfeed_enqueue

Enqueue a Tensor on the computation outfeed.

``` python
tf.contrib.tpu.outfeed_enqueue(
    input,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`input`</b>: A `Tensor`. A tensor that will be inserted into the outfeed queue.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The created Operation.
