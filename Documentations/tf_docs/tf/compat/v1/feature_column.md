description: Public API for tf._api.v2.feature_column namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.feature_column" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.feature_column

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.feature_column namespace



## Functions

[`bucketized_column(...)`](../../../tf/feature_column/bucketized_column.md): Represents discretized dense input bucketed by `boundaries`. (deprecated)

[`categorical_column_with_hash_bucket(...)`](../../../tf/feature_column/categorical_column_with_hash_bucket.md): Represents sparse feature where ids are set by hashing. (deprecated)

[`categorical_column_with_identity(...)`](../../../tf/feature_column/categorical_column_with_identity.md): A `CategoricalColumn` that returns identity values. (deprecated)

[`categorical_column_with_vocabulary_file(...)`](../../../tf/compat/v1/feature_column/categorical_column_with_vocabulary_file.md): A `CategoricalColumn` with a vocabulary file. (deprecated)

[`categorical_column_with_vocabulary_list(...)`](../../../tf/feature_column/categorical_column_with_vocabulary_list.md): A `CategoricalColumn` with in-memory vocabulary. (deprecated)

[`crossed_column(...)`](../../../tf/feature_column/crossed_column.md): Returns a column for performing crosses of categorical features. (deprecated)

[`embedding_column(...)`](../../../tf/feature_column/embedding_column.md): `DenseColumn` that converts from sparse, categorical input. (deprecated)

[`indicator_column(...)`](../../../tf/feature_column/indicator_column.md): Represents multi-hot representation of given categorical column. (deprecated)

[`input_layer(...)`](../../../tf/compat/v1/feature_column/input_layer.md): Returns a dense `Tensor` as input layer based on given `feature_columns`. (deprecated)

[`linear_model(...)`](../../../tf/compat/v1/feature_column/linear_model.md): Returns a linear prediction `Tensor` based on given `feature_columns`. (deprecated)

[`make_parse_example_spec(...)`](../../../tf/compat/v1/feature_column/make_parse_example_spec.md): Creates parsing spec dictionary from input feature_columns. (deprecated)

[`numeric_column(...)`](../../../tf/feature_column/numeric_column.md): Represents real valued or numerical features. (deprecated)

[`sequence_categorical_column_with_hash_bucket(...)`](../../../tf/feature_column/sequence_categorical_column_with_hash_bucket.md): A sequence of categorical terms where ids are set by hashing. (deprecated)

[`sequence_categorical_column_with_identity(...)`](../../../tf/feature_column/sequence_categorical_column_with_identity.md): Returns a feature column that represents sequences of integers. (deprecated)

[`sequence_categorical_column_with_vocabulary_file(...)`](../../../tf/feature_column/sequence_categorical_column_with_vocabulary_file.md): A sequence of categorical terms where ids use a vocabulary file. (deprecated)

[`sequence_categorical_column_with_vocabulary_list(...)`](../../../tf/feature_column/sequence_categorical_column_with_vocabulary_list.md): A sequence of categorical terms where ids use an in-memory list. (deprecated)

[`sequence_numeric_column(...)`](../../../tf/feature_column/sequence_numeric_column.md): Returns a feature column that represents sequences of numeric data. (deprecated)

[`shared_embedding_columns(...)`](../../../tf/compat/v1/feature_column/shared_embedding_columns.md): List of dense columns that convert from sparse, categorical input. (deprecated)

[`weighted_categorical_column(...)`](../../../tf/feature_column/weighted_categorical_column.md): Applies weight values to a `CategoricalColumn`. (deprecated)

