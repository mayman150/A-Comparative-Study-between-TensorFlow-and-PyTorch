description: Returns the element at a specific index in a datasest.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.at" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.at

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/random_access.py">View source</a>



Returns the element at a specific index in a datasest.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.at(
    dataset, index
)
</code></pre>



<!-- Placeholder for "Used in" -->

Currently, random access is supported for the following tf.data operations:

   - <a href="../../../tf/data/Dataset.md#from_tensor_slices"><code>tf.data.Dataset.from_tensor_slices</code></a>,
   - <a href="../../../tf/data/Dataset.md#from_tensors"><code>tf.data.Dataset.from_tensors</code></a>,
   - <a href="../../../tf/data/Dataset.md#shuffle"><code>tf.data.Dataset.shuffle</code></a>,
   - <a href="../../../tf/data/Dataset.md#batch"><code>tf.data.Dataset.batch</code></a>,
   - <a href="../../../tf/data/Dataset.md#shard"><code>tf.data.Dataset.shard</code></a>,
   - <a href="../../../tf/data/Dataset.md#map"><code>tf.data.Dataset.map</code></a>,
   - <a href="../../../tf/data/Dataset.md#range"><code>tf.data.Dataset.range</code></a>,
   - <a href="../../../tf/data/Dataset.md#zip"><code>tf.data.Dataset.zip</code></a>,
   - <a href="../../../tf/data/Dataset.md#skip"><code>tf.data.Dataset.skip</code></a>,
   - <a href="../../../tf/data/Dataset.md#repeat"><code>tf.data.Dataset.repeat</code></a>,
   - <a href="../../../tf/data/Dataset.md#list_files"><code>tf.data.Dataset.list_files</code></a>,
   - `tf.data.Dataset.SSTableDataset`,
   - <a href="../../../tf/data/Dataset.md#concatenate"><code>tf.data.Dataset.concatenate</code></a>,
   - <a href="../../../tf/data/Dataset.md#enumerate"><code>tf.data.Dataset.enumerate</code></a>,
   - `tf.data.Dataset.parallel_map`,
   - <a href="../../../tf/data/Dataset.md#prefetch"><code>tf.data.Dataset.prefetch</code></a>,
   - <a href="../../../tf/data/Dataset.md#take"><code>tf.data.Dataset.take</code></a>,
   - <a href="../../../tf/data/Dataset.md#cache"><code>tf.data.Dataset.cache</code></a> (in-memory only)

   Users can use the cache operation to enable random access for any dataset,
   even one comprised of transformations which are not on this list.
   E.g., to get the third element of a TFDS dataset:

     ```python
     ds = tfds.load("mnist", split="train").cache()
     elem = tf.data.Dataset.experimental.at(ds, 3)
     ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`<a id="dataset"></a>
</td>
<td>
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> to determine whether it supports random access.
</td>
</tr><tr>
<td>
`index`<a id="index"></a>
</td>
<td>
The index at which to fetch the element.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A (nested) structure of values matching <a href="../../../tf/data/Dataset.md#element_spec"><code>tf.data.Dataset.element_spec</code></a>.
</td>
</tr>
<tr>
<td>
`Raises`<a id="Raises"></a>
</td>
<td>
  UnimplementedError: If random access is not yet supported for a dataset.
</td>
</tr>
</table>

