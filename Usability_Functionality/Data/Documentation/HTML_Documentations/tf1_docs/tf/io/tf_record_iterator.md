<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.tf_record_iterator" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.tf_record_iterator

An iterator that read the records from a TFRecords file. (deprecated)

### Aliases:

* `tf.compat.v1.io.tf_record_iterator`
* `tf.compat.v1.python_io.tf_record_iterator`
* `tf.compat.v2.compat.v1.io.tf_record_iterator`
* `tf.compat.v2.compat.v1.python_io.tf_record_iterator`
* `tf.io.tf_record_iterator`
* `tf.python_io.tf_record_iterator`

``` python
tf.io.tf_record_iterator(
    path,
    options=None
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use eager execution and: 
<a href="../../tf/data/TFRecordDataset.md"><code>tf.data.TFRecordDataset(path)</code></a>

#### Args:


* <b>`path`</b>: The path to the TFRecords file.
* <b>`options`</b>: (optional) A TFRecordOptions object.


#### Yields:

Strings.



#### Raises:


* <b>`IOError`</b>: If `path` cannot be opened for reading.