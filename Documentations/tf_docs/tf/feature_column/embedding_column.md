description: DenseColumn that converts from sparse, categorical input. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.feature_column.embedding_column" />
<meta itemprop="path" content="Stable" />
</div>

# tf.feature_column.embedding_column

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column_v2.py">View source</a>



`DenseColumn` that converts from sparse, categorical input. (deprecated)



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
<p>`tf.compat.v1.feature_column.embedding_column`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.feature_column.embedding_column(
    categorical_column,
    dimension,
    combiner=&#x27;mean&#x27;,
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True,
    use_safe_embedding_lookup=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the <a href="../../tf/keras/utils/FeatureSpace.md"><code>tf.keras.utils.FeatureSpace</code></a> utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.

Use this when your inputs are sparse, but you want to convert them to a dense
representation (e.g., to feed to a DNN).

Inputs must be a `CategoricalColumn` created by any of the
`categorical_column_*` function. Here is an example of using
`embedding_column` with `DNNClassifier`:

```python
video_id = categorical_column_with_identity(
    key='video_id', num_buckets=1000000, default_value=0)
columns = [embedding_column(video_id, 9),...]

estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

label_column = ...
def input_fn():
  features = tf.io.parse_example(
      ..., features=make_parse_example_spec(columns + [label_column]))
  labels = features.pop(label_column.name)
  return features, labels

estimator.train(input_fn=input_fn, steps=100)
```

Here is an example using `embedding_column` with model_fn:

```python
def model_fn(features, ...):
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [embedding_column(video_id, 9),...]
  dense_tensor = input_layer(features, columns)
  # Form DNN layers, calculate loss, and return EstimatorSpec.
  ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`categorical_column`<a id="categorical_column"></a>
</td>
<td>
A `CategoricalColumn` created by a
`categorical_column_with_*` function. This column produces the sparse IDs
that are inputs to the embedding lookup.
</td>
</tr><tr>
<td>
`dimension`<a id="dimension"></a>
</td>
<td>
An integer specifying dimension of the embedding, must be > 0.
</td>
</tr><tr>
<td>
`combiner`<a id="combiner"></a>
</td>
<td>
A string specifying how to reduce if there are multiple entries in
a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
'mean' the default. 'sqrtn' often achieves good accuracy, in particular
with bag-of-words columns. Each of this can be thought as example level
normalizations on the column. For more information, see
`tf.embedding_lookup_sparse`.
</td>
</tr><tr>
<td>
`initializer`<a id="initializer"></a>
</td>
<td>
A variable initializer function to be used in embedding
variable initialization. If not specified, defaults to
`truncated_normal_initializer` with mean `0.0` and standard deviation
`1/sqrt(dimension)`.
</td>
</tr><tr>
<td>
`ckpt_to_load_from`<a id="ckpt_to_load_from"></a>
</td>
<td>
String representing checkpoint name/pattern from which to
restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
</td>
</tr><tr>
<td>
`tensor_name_in_ckpt`<a id="tensor_name_in_ckpt"></a>
</td>
<td>
Name of the `Tensor` in `ckpt_to_load_from` from which
to restore the column weights. Required if `ckpt_to_load_from` is not
`None`.
</td>
</tr><tr>
<td>
`max_norm`<a id="max_norm"></a>
</td>
<td>
If not `None`, embedding values are l2-normalized to this value.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
Whether or not the embedding is trainable. Default is True.
</td>
</tr><tr>
<td>
`use_safe_embedding_lookup`<a id="use_safe_embedding_lookup"></a>
</td>
<td>
If true, uses safe_embedding_lookup_sparse
instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
there are no empty rows and all weights and ids are positive at the
expense of extra compute cost. This only applies to rank 2 (NxM) shaped
input tensors. Defaults to true, consider turning off if the above checks
are not needed. Note that having empty rows will not trigger any error
though the output result might be 0 or omitted.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`DenseColumn` that converts from sparse input.
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
if `dimension` not > 0.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
is specified.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `initializer` is specified and is not callable.
</td>
</tr><tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
If eager execution is enabled.
</td>
</tr>
</table>

