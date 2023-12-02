<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.get_output_types" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.get_output_types

Returns the output shapes of a `Dataset` or `Iterator` elements.

### Aliases:

* `tf.compat.v1.data.get_output_types`
* `tf.compat.v2.compat.v1.data.get_output_types`
* `tf.data.get_output_types`

``` python
tf.data.get_output_types(dataset_or_iterator)
```

<!-- Placeholder for "Used in" -->

This utility method replaces the deprecated-in-V2
`tf.compat.v1.Dataset.output_types` property.

#### Args:


* <b>`dataset_or_iterator`</b>: A <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> or <a href="../../tf/data/Iterator.md"><code>tf.data.Iterator</code></a>.


#### Returns:

A nested structure of <a href="../../tf/dtypes/DType.md"><code>tf.DType</code></a> objects objects matching the structure of
dataset / iterator elements and specifying the shape of the individual
components.
