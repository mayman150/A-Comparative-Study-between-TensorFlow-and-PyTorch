<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.data.TFRecordWriter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="write"/>
</div>

# tf.contrib.data.TFRecordWriter

## Class `TFRecordWriter`

Writes data to a TFRecord file.

Inherits From: [`TFRecordWriter`](../../../tf/data/experimental/TFRecordWriter.md)

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    filename,
    compression_type=None
)
```

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/experimental/TFRecordWriter.md"><code>tf.data.experimental.TFRecordWriter(...)</code></a>.



## Methods

<h3 id="write"><code>write</code></h3>

``` python
write(dataset)
```

Returns a <a href="../../../tf/Operation.md"><code>tf.Operation</code></a> to write a dataset to a file.


#### Args:


* <b>`dataset`</b>: a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> whose elements are to be written to a file


#### Returns:

A <a href="../../../tf/Operation.md"><code>tf.Operation</code></a> that, when run, writes contents of `dataset` to a file.




