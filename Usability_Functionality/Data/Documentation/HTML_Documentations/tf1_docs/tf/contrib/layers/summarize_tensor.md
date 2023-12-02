<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.summarize_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.summarize_tensor

Summarize a tensor using a suitable summary type.

``` python
tf.contrib.layers.summarize_tensor(
    tensor,
    tag=None
)
```

<!-- Placeholder for "Used in" -->

This function adds a summary op for `tensor`. The type of summary depends on
the shape of `tensor`. For scalars, a `scalar_summary` is created, for all
other tensors, `histogram_summary` is used.

#### Args:


* <b>`tensor`</b>: The tensor to summarize
* <b>`tag`</b>: The tag to use, if None then use tensor's op's name.


#### Returns:

The summary op created or None for string tensors.
