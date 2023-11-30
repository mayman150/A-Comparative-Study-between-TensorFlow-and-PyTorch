description: Returns a dense Tensor as input layer based on given feature_columns. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.feature_column.input_layer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.feature_column.input_layer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/feature_column/feature_column.py">View source</a>



Returns a dense `Tensor` as input layer based on given `feature_columns`. (deprecated)



Warning: tf.feature_column is not recommended for new code. Instead,
feature preprocessing can be done directly using either [Keras preprocessing
layers](https://www.tensorflow.org/guide/migrate/migrating_feature_columns)
or through the one-stop utility [`tf.keras.utils.FeatureSpace`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/FeatureSpace)
built on top of them. See the [migration guide](https://tensorflow.org/guide/migrate)
for details.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.feature_column.input_layer(
    features,
    feature_columns,
    weight_collections=None,
    trainable=True,
    cols_to_vars=None,
    cols_to_output_tensors=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use Keras preprocessing layers instead, either directly or via the <a href="../../../../tf/keras/utils/FeatureSpace.md"><code>tf.keras.utils.FeatureSpace</code></a> utility. Each of `tf.feature_column.*` has a functional equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.

Generally a single example in training data is described with FeatureColumns.
At the first layer of the model, this column oriented data should be converted
to a single `Tensor`.

#### Example:



```python
price = numeric_column('price')
keywords_embedded = embedding_column(
    categorical_column_with_hash_bucket("keywords", 10K), dimensions=16)
columns = [price, keywords_embedded, ...]
features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
dense_tensor = input_layer(features, columns)
for units in [128, 64, 32]:
  dense_tensor = tf.compat.v1.layers.dense(dense_tensor, units, tf.nn.relu)
prediction = tf.compat.v1.layers.dense(dense_tensor, 1)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`features`<a id="features"></a>
</td>
<td>
A mapping from key to tensors. `_FeatureColumn`s look up via these
keys. For example `numeric_column('price')` will look at 'price' key in
this dict. Values can be a `SparseTensor` or a `Tensor` depends on
corresponding `_FeatureColumn`.
</td>
</tr><tr>
<td>
`feature_columns`<a id="feature_columns"></a>
</td>
<td>
An iterable containing the FeatureColumns to use as inputs
to your model. All items should be instances of classes derived from
`_DenseColumn` such as `numeric_column`, `embedding_column`,
`bucketized_column`, `indicator_column`. If you have categorical features,
you can wrap them with an `embedding_column` or `indicator_column`.
</td>
</tr><tr>
<td>
`weight_collections`<a id="weight_collections"></a>
</td>
<td>
A list of collection names to which the Variable will be
added. Note that variables will also be added to collections
`tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
If `True` also add the variable to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see <a href="../../../../tf/Variable.md"><code>tf.Variable</code></a>).
</td>
</tr><tr>
<td>
`cols_to_vars`<a id="cols_to_vars"></a>
</td>
<td>
If not `None`, must be a dictionary that will be filled with a
mapping from `_FeatureColumn` to list of `Variable`s.  For example, after
the call, we might have cols_to_vars = {_EmbeddingColumn(
categorical_column=_HashedCategoricalColumn( key='sparse_feature',
hash_bucket_size=5, dtype=tf.string), dimension=10): [<tf.Variable
'some_variable:0' shape=(5, 10), <tf.Variable 'some_variable:1' shape=(5,
10)]} If a column creates no variables, its value will be an empty list.
</td>
</tr><tr>
<td>
`cols_to_output_tensors`<a id="cols_to_output_tensors"></a>
</td>
<td>
If not `None`, must be a dictionary that will be
filled with a mapping from '_FeatureColumn' to the associated output
`Tensor`s.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` which represents input layer of a model. Its shape
is (batch_size, first_layer_dimension) and its dtype is `float32`.
first_layer_dimension is determined based on given `feature_columns`.
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
if an item in `feature_columns` is not a `_DenseColumn`.
</td>
</tr>
</table>

