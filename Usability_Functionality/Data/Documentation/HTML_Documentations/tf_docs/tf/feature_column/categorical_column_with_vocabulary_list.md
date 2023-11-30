description: A CategoricalColumn with in-memory vocabulary. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.categorical_column_with_vocabulary_list" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.categorical_column_with_vocabulary_list

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



A `CategoricalColumn` with in-memory vocabulary. (deprecated)



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
<p>`tf.compat.v1.feature_column.categorical_column_with_vocabulary_list`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.categorical_column_with_vocabulary_list(
    key, vocabulary_list, dtype=None, default_value=-1, num_oov_buckets=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the <a href="../../tf/keras/utils/FeatureSpace.md"><code>tf.keras.utils.FeatureSpace</code></a> utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.

Use this when your inputs are in string or integer format, and you have an
in-memory vocabulary mapping each value to an integer ID. By default,
out-of-vocabulary values are ignored. Use either (but not both) of
`num_oov_buckets` and `default_value` to specify how to include
out-of-vocabulary values.

For input dictionary `features`, `features[key]` is either `Tensor` or
`SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
and `''` for string, which will be dropped by this feature column.

Example with `num_oov_buckets`:
In the following example, each input in `vocabulary_list` is assigned an ID
0-3 corresponding to its index (e.g., input 'B' produces output 2). All other
inputs are hashed and assigned an ID 4-5.

```python
colors = categorical_column_with_vocabulary_list(
    key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
    num_oov_buckets=2)
columns = [colors, ...]
features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
linear_prediction, _, _ = linear_model(features, columns)
```

Example with `default_value`:
In the following example, each input in `vocabulary_list` is assigned an ID
0-4 corresponding to its index (e.g., input 'B' produces output 3). All other
inputs are assigned `default_value` 0.


```python
colors = categorical_column_with_vocabulary_list(
    key='colors', vocabulary_list=('X', 'R', 'G', 'B', 'Y'), default_value=0)
columns = [colors, ...]
features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
linear_prediction, _, _ = linear_model(features, columns)
```

And to make an embedding with either:

```python
columns = [embedding_column(colors, 3),...]
features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
dense_tensor = input_layer(features, columns)
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
`vocabulary_list`<a id="vocabulary_list"></a>
</td>
<td>
An ordered iterable defining the vocabulary. Each feature
is mapped to the index of its value (if present) in `vocabulary_list`.
Must be castable to `dtype`.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The type of features. Only string and integer types are supported. If
`None`, it will be inferred from `vocabulary_list`.
</td>
</tr><tr>
<td>
`default_value`<a id="default_value"></a>
</td>
<td>
The integer ID value to return for out-of-vocabulary feature
values, defaults to `-1`. This can not be specified with a positive
`num_oov_buckets`.
</td>
</tr><tr>
<td>
`num_oov_buckets`<a id="num_oov_buckets"></a>
</td>
<td>
Non-negative integer, the number of out-of-vocabulary
buckets. All out-of-vocabulary inputs will be assigned IDs in the range
`[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
hash of the input value. A positive `num_oov_buckets` can not be specified
with `default_value`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `CategoricalColumn` with in-memory vocabulary.
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
if `vocabulary_list` is empty, or contains duplicate keys.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
`num_oov_buckets` is a negative integer.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
`num_oov_buckets` and `default_value` are both specified.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `dtype` is not integer or string.
</td>
</tr>
</table>

