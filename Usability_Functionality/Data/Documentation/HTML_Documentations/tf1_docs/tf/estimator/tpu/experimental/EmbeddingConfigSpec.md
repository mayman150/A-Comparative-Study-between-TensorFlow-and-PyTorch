<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.estimator.tpu.experimental.EmbeddingConfigSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="feature_columns"/>
<meta itemprop="property" content="optimization_parameters"/>
<meta itemprop="property" content="clipping_limit"/>
<meta itemprop="property" content="pipeline_execution_with_tensor_core"/>
<meta itemprop="property" content="experimental_gradient_multiplier_fn"/>
<meta itemprop="property" content="feature_to_config_dict"/>
<meta itemprop="property" content="table_to_config_dict"/>
<meta itemprop="property" content="partition_strategy"/>
</div>

# tf.estimator.tpu.experimental.EmbeddingConfigSpec

## Class `EmbeddingConfigSpec`

Class to keep track of the specification for TPU embeddings.



### Aliases:

* Class `tf.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec`
* Class `tf.compat.v2.compat.v1.estimator.tpu.experimental.EmbeddingConfigSpec`
* Class `tf.estimator.tpu.experimental.EmbeddingConfigSpec`



Defined in [`python/estimator/tpu/_tpu_estimator_embedding.py`](https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/tpu/_tpu_estimator_embedding.py).

<!-- Placeholder for "Used in" -->

Pass this class to <a href="../../../../tf/estimator/tpu/TPUEstimator.md"><code>tf.estimator.tpu.TPUEstimator</code></a> via the
`embedding_config_spec` parameter. At minimum you need to specify
`feature_columns` and `optimization_parameters`. The feature columns passed
should be created with some combination of
<a href="../../../../tf/tpu/experimental/embedding_column.md"><code>tf.tpu.experimental.embedding_column</code></a> and
<a href="../../../../tf/tpu/experimental/shared_embedding_columns.md"><code>tf.tpu.experimental.shared_embedding_columns</code></a>.

TPU embeddings do not support arbitrary Tensorflow optimizers and the
main optimizer you use for your model will be ignored for the embedding table
variables. Instead TPU embeddigns support a fixed set of predefined optimizers
that you can select from and set the parameters of. These include adagrad,
adam and stochastic gradient descent. Each supported optimizer has a
`Parameters` class in the <a href="../../../../tf/tpu/experimental.md"><code>tf.tpu.experimental</code></a> namespace.

```
column_a = tf.feature_column.categorical_column_with_identity(...)
column_b = tf.feature_column.categorical_column_with_identity(...)
column_c = tf.feature_column.categorical_column_with_identity(...)
tpu_shared_columns = tf.tpu.experimental.shared_embedding_columns(
    [column_a, column_b], 10)
tpu_non_shared_column = tf.tpu.experimental.embedding_column(
    column_c, 10)
tpu_columns = [tpu_non_shared_column] + tpu_shared_columns
...
def model_fn(features):
  dense_features = tf.keras.layers.DenseFeature(tpu_columns)
  embedded_feature = dense_features(features)
  ...

estimator = tf.estimator.tpu.TPUEstimator(
    model_fn=model_fn,
    ...
    embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        column=tpu_columns,
        optimization_parameters=(
            tf.estimator.tpu.experimental.AdagradParameters(0.1))))

## Properties

<h3 id="feature_columns"><code>feature_columns</code></h3>




<h3 id="optimization_parameters"><code>optimization_parameters</code></h3>




<h3 id="clipping_limit"><code>clipping_limit</code></h3>




<h3 id="pipeline_execution_with_tensor_core"><code>pipeline_execution_with_tensor_core</code></h3>




<h3 id="experimental_gradient_multiplier_fn"><code>experimental_gradient_multiplier_fn</code></h3>




<h3 id="feature_to_config_dict"><code>feature_to_config_dict</code></h3>




<h3 id="table_to_config_dict"><code>table_to_config_dict</code></h3>




<h3 id="partition_strategy"><code>partition_strategy</code></h3>






