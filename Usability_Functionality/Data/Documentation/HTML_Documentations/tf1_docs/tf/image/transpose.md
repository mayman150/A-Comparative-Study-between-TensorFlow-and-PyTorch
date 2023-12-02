<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.image.transpose" />
<meta itemprop="path" content="Stable" />
</div>

# tf.image.transpose

Transpose image(s) by swapping the height and width dimension.

### Aliases:

* `tf.compat.v1.image.transpose`
* `tf.compat.v1.image.transpose_image`
* `tf.compat.v2.compat.v1.image.transpose`
* `tf.compat.v2.compat.v1.image.transpose_image`
* `tf.compat.v2.image.transpose`
* `tf.image.transpose`
* `tf.image.transpose_image`

``` python
tf.image.transpose(
    image,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`image`</b>: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
  of shape `[height, width, channels]`.
* <b>`name`</b>: A name for this operation (optional).


#### Returns:

 If `image` was 4-D, a 4-D float Tensor of shape
`[batch, width, height, channels]`
 If `image` was 3-D, a 3-D float Tensor of shape
`[width, height, channels]`



#### Raises:


* <b>`ValueError`</b>: if the shape of `image` not supported.