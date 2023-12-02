<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.summarize_activation" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.summarize_activation

Summarize an activation.

``` python
tf.contrib.layers.summarize_activation(op)
```

<!-- Placeholder for "Used in" -->

This applies the given activation and adds useful summaries specific to the
activation.

#### Args:


* <b>`op`</b>: The tensor to summarize (assumed to be a layer activation).

#### Returns:

The summary op created to summarize `op`.
