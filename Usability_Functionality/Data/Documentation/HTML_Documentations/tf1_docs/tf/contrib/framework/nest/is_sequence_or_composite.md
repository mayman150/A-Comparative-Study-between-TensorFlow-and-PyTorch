<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.nest.is_sequence_or_composite" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.nest.is_sequence_or_composite

Returns true if its input is a sequence or a `CompositeTensor`.

``` python
tf.contrib.framework.nest.is_sequence_or_composite(o)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`seq`</b>: an input sequence.


#### Returns:

True if the sequence is a not a string and is a collections.Sequence or a
dict or a CompositeTensor or a TypeSpec (except string and TensorSpec).
