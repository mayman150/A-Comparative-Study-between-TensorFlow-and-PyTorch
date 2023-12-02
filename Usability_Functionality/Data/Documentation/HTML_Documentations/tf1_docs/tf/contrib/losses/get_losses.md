<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.losses.get_losses" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.losses.get_losses

Gets the list of losses from the loss_collection. (deprecated)

``` python
tf.contrib.losses.get_losses(
    scope=None,
    loss_collection=tf.GraphKeys.LOSSES
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-30.
Instructions for updating:
Use tf.losses.get_losses instead.

#### Args:


* <b>`scope`</b>: an optional scope for filtering the losses to return.
* <b>`loss_collection`</b>: Optional losses collection.


#### Returns:

a list of loss tensors.
