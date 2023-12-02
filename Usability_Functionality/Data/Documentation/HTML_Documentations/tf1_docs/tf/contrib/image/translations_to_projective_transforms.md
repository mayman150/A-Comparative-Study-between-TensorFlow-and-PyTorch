<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.image.translations_to_projective_transforms" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.image.translations_to_projective_transforms

Returns projective transform(s) for the given translation(s).

``` python
tf.contrib.image.translations_to_projective_transforms(
    translations,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`translations`</b>: A 2-element list representing [dx, dy] or a matrix of
    2-element lists representing [dx, dy] to translate for each image
    (for a batch of images). The rank must be statically known (the shape
    is not `TensorShape(None)`.
* <b>`name`</b>: The name of the op.


#### Returns:

A tensor of shape (num_images, 8) projective transforms which can be given
    to <a href="../../../tf/contrib/image/transform.md"><code>tf.contrib.image.transform</code></a>.
