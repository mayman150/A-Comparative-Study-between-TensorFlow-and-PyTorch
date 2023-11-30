description: Loads a previously saved dataset. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.load" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.load

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/io.py">View source</a>



Loads a previously saved dataset. (deprecated)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.load(
    path, element_spec=None, compression=None, reader_func=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/Dataset.md#load"><code>tf.data.Dataset.load(...)</code></a> instead.

#### Example usage:



```
>>> import tempfile
>>> path = os.path.join(tempfile.gettempdir(), "saved_data")
>>> # Save a dataset
>>> dataset = tf.data.Dataset.range(2)
>>> tf.data.experimental.save(dataset, path)
>>> new_dataset = tf.data.experimental.load(path)
>>> for elem in new_dataset:
...   print(elem)
tf.Tensor(0, shape=(), dtype=int64)
tf.Tensor(1, shape=(), dtype=int64)
```


If the default option of sharding the saved dataset was used, the element
order of the saved dataset will be preserved when loading it.

The `reader_func` argument can be used to specify a custom order in which
elements should be loaded from the individual shards. The `reader_func` is
expected to take a single argument -- a dataset of datasets, each containing
elements of one of the shards -- and return a dataset of elements. For
example, the order of shards can be shuffled when loading them as follows:

```python
def custom_reader_func(datasets):
  datasets = datasets.shuffle(NUM_SHARDS)
  return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

dataset = tf.data.experimental.load(
    path="/path/to/data", ..., reader_func=custom_reader_func)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`<a id="path"></a>
</td>
<td>
Required. A path pointing to a previously saved dataset.
</td>
</tr><tr>
<td>
`element_spec`<a id="element_spec"></a>
</td>
<td>
Optional. A nested structure of <a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> objects matching
the structure of an element of the saved dataset and specifying the type
of individual element components. If not provided, the nested structure of
<a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> saved with the saved dataset is used. Note that this
argument is required in graph mode.
</td>
</tr><tr>
<td>
`compression`<a id="compression"></a>
</td>
<td>
Optional. The algorithm to use to decompress the data when
reading it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
</td>
</tr><tr>
<td>
`reader_func`<a id="reader_func"></a>
</td>
<td>
Optional. A function to control how to read data from shards.
If present, the function will be traced and executed as graph computation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`FileNotFoundError`<a id="FileNotFoundError"></a>
</td>
<td>
If `element_spec` is not specified and the saved nested
structure of <a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> can not be located with the saved dataset.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If `element_spec` is not specified and the method is executed
in graph mode.
</td>
</tr>
</table>

