description: Split a dataset into a left half and a right half (e.g. train / test).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.split_dataset" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.split_dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/dataset_utils.py#L32-L117">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Split a dataset into a left half and a right half (e.g. train / test).


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.split_dataset(
    dataset, left_size=None, right_size=None, shuffle=False, seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`<a id="dataset"></a>
</td>
<td>
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object, or a list/tuple of arrays with the
same length.
</td>
</tr><tr>
<td>
`left_size`<a id="left_size"></a>
</td>
<td>
If float (in the range `[0, 1]`), it signifies
the fraction of the data to pack in the left dataset. If integer, it
signifies the number of samples to pack in the left dataset. If
`None`, it uses the complement to `right_size`. Defaults to `None`.
</td>
</tr><tr>
<td>
`right_size`<a id="right_size"></a>
</td>
<td>
If float (in the range `[0, 1]`), it signifies
the fraction of the data to pack in the right dataset. If integer, it
signifies the number of samples to pack in the right dataset. If
`None`, it uses the complement to `left_size`. Defaults to `None`.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Boolean, whether to shuffle the data before splitting it.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A random seed for shuffling.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of two <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> objects: the left and right splits.
</td>
</tr>

</table>



#### Example:



```
>>> data = np.random.random(size=(1000, 4))
>>> left_ds, right_ds = tf.keras.utils.split_dataset(data, left_size=0.8)
>>> int(left_ds.cardinality())
800
>>> int(right_ds.cardinality())
200
```