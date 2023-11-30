description: A CategoricalColumn that returns identity values. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.categorical_column_with_identity" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.categorical_column_with_identity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



A `CategoricalColumn` that returns identity values. (deprecated)



Warning: tf.feature_column is not recommended for new code. Instead,
feature preprocessing can be done directly using either [Keras preprocessing
layers](https://www.tensorflow.org/guide/migrate/migrating_feature_columns)
or through the one-stop utility [`tf.keras.utils.FeatureSpace`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/FeatureSpace)
built on top of them. See the [migration guide](https://tensorflow.org/guide/migrate)
for details.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.feature_column.categorical_column_with_identity`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.categorical_column_with_identity(
    key, num_buckets, default_value=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the <a href="../../tf/keras/utils/FeatureSpace.md"><code>tf.keras.utils.FeatureSpace</code></a> utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.

Use this when your inputs are integers in the range `[0, num_buckets)`, and
you want to use the input value itself as the categorical ID. Values outside
this range will result in `default_value` if specified, otherwise it will
fail.

Typically, this is used for contiguous ranges of integer indexes, but
it doesn't have to be. This might be inefficient, however, if many of IDs
are unused. Consider `categorical_column_with_hash_bucket` in that case.

For input dictionary `features`, `features[key]` is either `Tensor` or
`SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
and `''` for string, which will be dropped by this feature column.

In the following examples, each input in the range `[0, 1000000)` is assigned
the same value. All other inputs are assigned `default_value` 0. Note that a
literal 0 in inputs will result in the same default ID.

#### Linear model:



```python
import tensorflow as tf
video_id = tf.feature_column.categorical_column_with_identity(
    key='video_id', num_buckets=1000000, default_value=0)
columns = [video_id]
features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
[33,78, 2, 73, 1]])}
linear_prediction = tf.compat.v1.feature_column.linear_model(features,
columns)
```

Embedding for a DNN model:

```python
import tensorflow as tf
video_id = tf.feature_column.categorical_column_with_identity(
    key='video_id', num_buckets=1000000, default_value=0)
columns = [tf.feature_column.embedding_column(video_id, 9)]
features = {'video_id': tf.sparse.from_dense([[2, 85, 0, 0, 0],
[33,78, 2, 73, 1]])}
input_layer = tf.keras.layers.DenseFeatures(columns)
dense_tensor = input_layer(features)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A unique string identifying the input feature. It is used as the column
name and the dictionary key for feature parsing configs, feature `Tensor`
objects, and feature columns.
</td>
</tr><tr>
<td>
`num_buckets`<a id="num_buckets"></a>
</td>
<td>
Range of inputs and outputs is `[0, num_buckets)`.
</td>
</tr><tr>
<td>
`default_value`<a id="default_value"></a>
</td>
<td>
If set, values outside of range `[0, num_buckets)` will be
replaced with this value. If not set, values >= num_buckets will cause a
failure while values < 0 will be dropped.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `CategoricalColumn` that returns identity values.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `num_buckets` is less than one.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `default_value` is not in range `[0, num_buckets)`.
</td>
</tr>
</table>

