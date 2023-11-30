description: Public API for tf._api.v2.data.experimental namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="AUTOTUNE"/>
<meta itemprop="property" content="INFINITE_CARDINALITY"/>
<meta itemprop="property" content="SHARD_HINT"/>
<meta itemprop="property" content="UNKNOWN_CARDINALITY"/>
</div>

# Module: tf.data.experimental

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.data.experimental namespace



## Modules

[`service`](../../tf/data/experimental/service.md) module: Public API for tf._api.v2.data.experimental.service namespace

## Classes

[`class AutoShardPolicy`](../../tf/data/experimental/AutoShardPolicy.md): Represents the type of auto-sharding to use.

[`class AutotuneAlgorithm`](../../tf/data/experimental/AutotuneAlgorithm.md): Represents the type of autotuning algorithm to use.

[`class AutotuneOptions`](../../tf/data/experimental/AutotuneOptions.md): Represents options for autotuning dataset performance.

[`class CheckpointInputPipelineHook`](../../tf/data/experimental/CheckpointInputPipelineHook.md): Checkpoints input pipeline state every N steps or seconds.

[`class CsvDataset`](../../tf/data/experimental/CsvDataset.md): A Dataset comprising lines from one or more CSV files.

[`class DatasetInitializer`](../../tf/data/experimental/DatasetInitializer.md): Creates a table initializer from a <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.

[`class DistributeOptions`](../../tf/data/experimental/DistributeOptions.md): Represents options for distributed data processing.

[`class ExternalStatePolicy`](../../tf/data/experimental/ExternalStatePolicy.md): Represents how to handle external state during serialization.

[`class OptimizationOptions`](../../tf/data/experimental/OptimizationOptions.md): Represents options for dataset optimizations.

[`class Optional`](../../tf/experimental/Optional.md): Represents a value that may or may not be present.

[`class RandomDataset`](../../tf/data/experimental/RandomDataset.md): A `Dataset` of pseudorandom values. (deprecated)

[`class Reducer`](../../tf/data/experimental/Reducer.md): A reducer is used for reducing a set of elements.

[`class SqlDataset`](../../tf/data/experimental/SqlDataset.md): A `Dataset` consisting of the results from a SQL query.

[`class TFRecordWriter`](../../tf/data/experimental/TFRecordWriter.md): Writes a dataset to a TFRecord file. (deprecated)

[`class ThreadingOptions`](../../tf/data/ThreadingOptions.md): Represents options for dataset threading.

## Functions

[`Counter(...)`](../../tf/data/experimental/Counter.md): Creates a `Dataset` that counts from `start` in steps of size `step`. (deprecated)

[`assert_cardinality(...)`](../../tf/data/experimental/assert_cardinality.md): Asserts the cardinality of the input dataset.

[`at(...)`](../../tf/data/experimental/at.md): Returns the element at a specific index in a datasest.

[`bucket_by_sequence_length(...)`](../../tf/data/experimental/bucket_by_sequence_length.md): A transformation that buckets elements in a `Dataset` by length. (deprecated)

[`cardinality(...)`](../../tf/data/experimental/cardinality.md): Returns the cardinality of `dataset`, if known.

[`choose_from_datasets(...)`](../../tf/data/experimental/choose_from_datasets.md): Creates a dataset that deterministically chooses elements from `datasets`. (deprecated)

[`copy_to_device(...)`](../../tf/data/experimental/copy_to_device.md): A transformation that copies dataset elements to the given `target_device`.

[`dense_to_ragged_batch(...)`](../../tf/data/experimental/dense_to_ragged_batch.md): A transformation that batches ragged elements into <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>s. (deprecated)

[`dense_to_sparse_batch(...)`](../../tf/data/experimental/dense_to_sparse_batch.md): A transformation that batches ragged elements into <a href="../../tf/sparse/SparseTensor.md"><code>tf.sparse.SparseTensor</code></a>s. (deprecated)

[`enable_debug_mode(...)`](../../tf/data/experimental/enable_debug_mode.md): Enables debug mode for tf.data.

[`enumerate_dataset(...)`](../../tf/data/experimental/enumerate_dataset.md): A transformation that enumerates the elements of a dataset. (deprecated)

[`from_list(...)`](../../tf/data/experimental/from_list.md): Creates a `Dataset` comprising the given list of elements.

[`from_variant(...)`](../../tf/data/experimental/from_variant.md): Constructs a dataset from the given variant and (nested) structure.

[`get_next_as_optional(...)`](../../tf/data/experimental/get_next_as_optional.md): Returns a <a href="../../tf/experimental/Optional.md"><code>tf.experimental.Optional</code></a> with the next element of the iterator. (deprecated)

[`get_single_element(...)`](../../tf/data/experimental/get_single_element.md): Returns the single element of the `dataset` as a nested structure of tensors. (deprecated)

[`get_structure(...)`](../../tf/data/experimental/get_structure.md): Returns the type signature for elements of the input dataset / iterator.

[`group_by_reducer(...)`](../../tf/data/experimental/group_by_reducer.md): A transformation that groups elements and performs a reduction.

[`group_by_window(...)`](../../tf/data/experimental/group_by_window.md): A transformation that groups windows of elements by key and reduces them. (deprecated)

[`ignore_errors(...)`](../../tf/data/experimental/ignore_errors.md): Creates a `Dataset` from another `Dataset` and silently ignores any errors. (deprecated)

[`index_table_from_dataset(...)`](../../tf/data/experimental/index_table_from_dataset.md): Returns an index lookup table based on the given dataset.

[`load(...)`](../../tf/data/experimental/load.md): Loads a previously saved dataset. (deprecated)

[`make_batched_features_dataset(...)`](../../tf/data/experimental/make_batched_features_dataset.md): Returns a `Dataset` of feature dictionaries from `Example` protos.

[`make_csv_dataset(...)`](../../tf/data/experimental/make_csv_dataset.md): Reads CSV files into a dataset.

[`make_saveable_from_iterator(...)`](../../tf/data/experimental/make_saveable_from_iterator.md): Returns a SaveableObject for saving/restoring iterator state using Saver. (deprecated)

[`map_and_batch(...)`](../../tf/data/experimental/map_and_batch.md): Fused implementation of `map` and `batch`. (deprecated)

[`pad_to_cardinality(...)`](../../tf/data/experimental/pad_to_cardinality.md): Pads a dataset with fake elements to reach the desired cardinality.

[`parallel_interleave(...)`](../../tf/data/experimental/parallel_interleave.md): A parallel version of the <a href="../../tf/data/Dataset.md#interleave"><code>Dataset.interleave()</code></a> transformation. (deprecated)

[`parse_example_dataset(...)`](../../tf/data/experimental/parse_example_dataset.md): A transformation that parses `Example` protos into a `dict` of tensors. (deprecated)

[`prefetch_to_device(...)`](../../tf/data/experimental/prefetch_to_device.md): A transformation that prefetches dataset values to the given `device`.

[`rejection_resample(...)`](../../tf/data/experimental/rejection_resample.md): A transformation that resamples a dataset to achieve a target distribution. (deprecated)

[`sample_from_datasets(...)`](../../tf/data/experimental/sample_from_datasets.md): Samples elements at random from the datasets in `datasets`. (deprecated)

[`save(...)`](../../tf/data/experimental/save.md): Saves the content of the given dataset. (deprecated)

[`scan(...)`](../../tf/data/experimental/scan.md): A transformation that scans a function across an input dataset. (deprecated)

[`shuffle_and_repeat(...)`](../../tf/data/experimental/shuffle_and_repeat.md): Shuffles and repeats a Dataset, reshuffling with each repetition. (deprecated)

[`snapshot(...)`](../../tf/data/experimental/snapshot.md): API to persist the output of the input dataset. (deprecated)

[`table_from_dataset(...)`](../../tf/data/experimental/table_from_dataset.md): Returns a lookup table based on the given dataset.

[`take_while(...)`](../../tf/data/experimental/take_while.md): A transformation that stops dataset iteration based on a `predicate`. (deprecated)

[`to_variant(...)`](../../tf/data/experimental/to_variant.md): Returns a variant representing the given dataset.

[`unbatch(...)`](../../tf/data/experimental/unbatch.md): Splits elements of a dataset into multiple elements on the batch dimension. (deprecated)

[`unique(...)`](../../tf/data/experimental/unique.md): Creates a `Dataset` from another `Dataset`, discarding duplicates. (deprecated)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
AUTOTUNE<a id="AUTOTUNE"></a>
</td>
<td>
`-1`
</td>
</tr><tr>
<td>
INFINITE_CARDINALITY<a id="INFINITE_CARDINALITY"></a>
</td>
<td>
`-1`
</td>
</tr><tr>
<td>
SHARD_HINT<a id="SHARD_HINT"></a>
</td>
<td>
`-1`
</td>
</tr><tr>
<td>
UNKNOWN_CARDINALITY<a id="UNKNOWN_CARDINALITY"></a>
</td>
<td>
`-2`
</td>
</tr>
</table>

