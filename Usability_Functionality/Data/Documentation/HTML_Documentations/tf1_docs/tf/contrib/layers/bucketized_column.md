<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.bucketized_column" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.bucketized_column

Creates a _BucketizedColumn for discretizing dense input.

``` python
tf.contrib.layers.bucketized_column(
    source_column,
    boundaries
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`source_column`</b>: A _RealValuedColumn defining dense column.
* <b>`boundaries`</b>: A list or tuple of floats specifying the boundaries. It has to
  be sorted.


#### Returns:

A _BucketizedColumn.



#### Raises:


* <b>`ValueError`</b>: if 'boundaries' is empty or not sorted.