<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.make_one_shot_iterator" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.make_one_shot_iterator

Creates a <a href="../../tf/data/Iterator.md"><code>tf.compat.v1.data.Iterator</code></a> for enumerating the elements of a dataset.

### Aliases:

* `tf.compat.v1.data.make_one_shot_iterator`
* `tf.compat.v2.compat.v1.data.make_one_shot_iterator`
* `tf.data.make_one_shot_iterator`

``` python
tf.data.make_one_shot_iterator(dataset)
```

<!-- Placeholder for "Used in" -->

Note: The returned iterator will be initialized automatically.
A "one-shot" iterator does not support re-initialization.

#### Args:


* <b>`dataset`</b>: A <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.


#### Returns:

A <a href="../../tf/data/Iterator.md"><code>tf.compat.v1.data.Iterator</code></a> over the elements of this dataset.
