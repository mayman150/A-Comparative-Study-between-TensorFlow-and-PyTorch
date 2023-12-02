<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.learn.multi_head" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.learn.multi_head

Creates a MultiHead stemming from same logits/hidden layer. (deprecated)

``` python
tf.contrib.learn.multi_head(
    heads,
    loss_weights=None
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

#### Args:


* <b>`heads`</b>: list of Head objects.
* <b>`loss_weights`</b>: optional list of weights to be used to merge losses from
    each head. All losses are weighted equally if not provided.


#### Returns:

A instance of `Head` that merges multiple heads.



#### Raises:


* <b>`ValueError`</b>: if heads and loss_weights have different size.