description: Pads a dataset with fake elements to reach the desired cardinality.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.pad_to_cardinality" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.pad_to_cardinality

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/pad_to_cardinality.py">View source</a>



Pads a dataset with fake elements to reach the desired cardinality.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.pad_to_cardinality`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.pad_to_cardinality(
    cardinality, mask_key=&#x27;valid&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

The dataset to pad must have a known and finite cardinality and contain
dictionary elements. The `mask_key` will be added to differentiate between
real and padding elements -- real elements will have a `<mask_key>=True` entry
while padding elements will have a `<mask_key>=False` entry.

#### Example usage:



```
>>> ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2]})
>>> ds = ds.apply(tf.data.experimental.pad_to_cardinality(3))
>>> list(ds.as_numpy_iterator())
[{'a': 1, 'valid': True}, {'a': 2, 'valid': True}, {'a': 0, 'valid': False}]
```

This can be useful, e.g. during eval, when partial batches are undesirable but
it is also important not to drop any data.

```
ds = ...
# Round up to the next full batch.
target_cardinality = -(-ds.cardinality() // batch_size) * batch_size
ds = ds.apply(tf.data.experimental.pad_to_cardinality(target_cardinality))
# Set `drop_remainder` so that batch shape will be known statically. No data
# will actually be dropped since the batch size divides the cardinality.
ds = ds.batch(batch_size, drop_remainder=True)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cardinality`<a id="cardinality"></a>
</td>
<td>
The cardinality to pad the dataset to.
</td>
</tr><tr>
<td>
`mask_key`<a id="mask_key"></a>
</td>
<td>
The key to use for identifying real vs padding elements.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dataset transformation that can be applied via <a href="../../../tf/data/Dataset.md#apply"><code>Dataset.apply()</code></a>.
</td>
</tr>

</table>

