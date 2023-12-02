<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.seq2seq.hardmax" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.seq2seq.hardmax

Returns batched one-hot vectors.

``` python
tf.contrib.seq2seq.hardmax(
    logits,
    name=None
)
```

<!-- Placeholder for "Used in" -->

The depth index containing the `1` is that of the maximum logit value.

#### Args:


* <b>`logits`</b>: A batch tensor of logit values.
* <b>`name`</b>: Name to use when creating ops.


#### Returns:

A batched one-hot tensor.
