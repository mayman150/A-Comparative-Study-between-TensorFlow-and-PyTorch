<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.get_output_shapes" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.get_output_shapes

Returns the output shapes of a `Dataset` or `Iterator` elements.

### Aliases:

* `tf.compat.v1.data.get_output_shapes`
* `tf.compat.v2.compat.v1.data.get_output_shapes`
* `tf.data.get_output_shapes`

``` python
tf.data.get_output_shapes(dataset_or_iterator)
```

<!-- Placeholder for "Used in" -->

This utility method replaces the deprecated-in-V2
`tf.compat.v1.Dataset.output_shapes` property.

#### Args:


* <b>`dataset_or_iterator`</b>: A <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> or <a href="../../tf/data/Iterator.md"><code>tf.data.Iterator</code></a>.


#### Returns:

A nested structure of <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> objects matching the structure of
the dataset / iterator elements and specifying the shape of the individual
components.
