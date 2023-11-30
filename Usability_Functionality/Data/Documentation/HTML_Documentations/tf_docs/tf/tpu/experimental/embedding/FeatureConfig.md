description: Configuration data for one embedding feature.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.embedding.FeatureConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.tpu.experimental.embedding.FeatureConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_v2_utils.py">View source</a>



Configuration data for one embedding feature.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.embedding.FeatureConfig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.embedding.FeatureConfig(
    table: <a href="../../../../tf/tpu/experimental/embedding/TableConfig.md"><code>tf.tpu.experimental.embedding.TableConfig</code></a>,
    max_sequence_length: int = 0,
    validate_weights_and_indices: bool = True,
    output_shape: Optional[Union[List[int], <a href="../../../../tf/TensorShape.md"><code>tf.TensorShape</code></a>]] = None,
    name: Optional[Text] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class holds the configuration data for a single embedding feature. The
main use is to assign features to <a href="../../../../tf/tpu/experimental/embedding/TableConfig.md"><code>tf.tpu.experimental.embedding.TableConfig</code></a>s
via the table parameter:

```python
table_config_one = tf.tpu.experimental.embedding.TableConfig(
    vocabulary_size=...,
    dim=...)
table_config_two = tf.tpu.experimental.embedding.TableConfig(
    vocabulary_size=...,
    dim=...)
feature_config = {
    'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config_one),
    'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config_one),
    'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config_two)}
embedding = tf.tpu.experimental.embedding.TPUEmbedding(
    feature_config=feature_config,
    batch_size=...
    optimizer=tf.tpu.experimental.embedding.Adam(0.1))
```

The above configuration has 2 tables, and three features. The first two
features will be looked up in the first table and the third feature will be
looked up in the second table.

You can also specify the output shape for each feature. The output shape
should be the expected activation shape excluding the table dimension. For
dense and sparse tensor, the output shape should be the same as the input
shape excluding the last dimension. For ragged tensor, the output shape can
mismatch the input shape.

NOTE: The `max_sequence_length` will be only used when the input tensor has
rank 2 and the `output_shape` is not set in the feature config.

When feeding features into `embedding.enqueue` they can be <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>s,
<a href="../../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>s or <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>s. When the argument
`max_sequence_length` is 0, the default, you should expect a output of
`embedding.dequeue` for this feature of shape `(batch_size, dim)`. If
`max_sequence_length` is greater than 0, the feature is embedded as a sequence
and padded up to the given length. The shape of the output for this feature
will be `(batch_size, max_sequence_length, dim)`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`table`<a id="table"></a>
</td>
<td>
An instance of <a href="../../../../tf/tpu/experimental/embedding/TableConfig.md"><code>tf.tpu.experimental.embedding.TableConfig</code></a>,
describing the table in which this feature should be looked up.
</td>
</tr><tr>
<td>
`max_sequence_length`<a id="max_sequence_length"></a>
</td>
<td>
If positive, the feature is a sequence feature with
the corresponding maximum sequence length. If the sequence is longer
than this, it will be truncated. If 0, the feature is not a sequence
feature.
</td>
</tr><tr>
<td>
`validate_weights_and_indices`<a id="validate_weights_and_indices"></a>
</td>
<td>
If true, uses safe_embedding_lookup during
serving which ensures there are no empty rows and all weights and ids
are positive at the expense of extra compute cost.
</td>
</tr><tr>
<td>
`output_shape`<a id="output_shape"></a>
</td>
<td>
Optional argument to config the output shape of the feature
activation. If provided, the feature feeding to the `embedding.enqueue`
has to match the shape (for ragged tensor, the input shape and output
shape can mismatch). If not provided, the shape can be either provided
to the `embedding.build` or auto detected at the runtime.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
An optional string used to name the table. Must be defined if
running on SparseCore.
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
if `table` is not an instance of
<a href="../../../../tf/tpu/experimental/embedding/TableConfig.md"><code>tf.tpu.experimental.embedding.TableConfig</code></a>.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `max_sequence_length` not an integer or is negative.
</td>
</tr>
</table>



