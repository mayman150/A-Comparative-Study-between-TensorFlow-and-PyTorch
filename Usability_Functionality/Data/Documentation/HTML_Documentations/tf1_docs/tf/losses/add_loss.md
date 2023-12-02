<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.losses.add_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tf.losses.add_loss

Adds a externally defined loss to the collection of losses.

### Aliases:

* `tf.compat.v1.losses.add_loss`
* `tf.compat.v2.compat.v1.losses.add_loss`
* `tf.losses.add_loss`

``` python
tf.losses.add_loss(
    loss,
    loss_collection=tf.GraphKeys.LOSSES
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`loss`</b>: A loss `Tensor`.
* <b>`loss_collection`</b>: Optional collection to add the loss to.