<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.lookup" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="FastHashSpec"/>
</div>

# Module: tf.contrib.lookup

Ops for lookup operations.

<!-- Placeholder for "Used in" -->



## Classes

[`class HashTable`](../../tf/contrib/lookup/HashTable.md): A generic hash table implementation.

[`class HasherSpec`](../../tf/contrib/lookup/HasherSpec.md): A structure for the spec of the hashing function to use for hash buckets.

[`class IdTableWithHashBuckets`](../../tf/contrib/lookup/IdTableWithHashBuckets.md): String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

[`class InitializableLookupTableBase`](../../tf/contrib/lookup/InitializableLookupTableBase.md): Initializable lookup table interface.

[`class KeyValueTensorInitializer`](../../tf/lookup/KeyValueTensorInitializer.md): Table initializers given `keys` and `values` tensors.

[`class LookupInterface`](../../tf/contrib/lookup/LookupInterface.md): Represent a lookup table that persists across different steps.

[`class MutableDenseHashTable`](../../tf/lookup/experimental/DenseHashTable.md): A generic mutable hash table implementation using tensors as backing store.

[`class MutableHashTable`](../../tf/contrib/lookup/MutableHashTable.md): A generic mutable hash table implementation.

[`class StrongHashSpec`](../../tf/contrib/lookup/StrongHashSpec.md): A structure to specify a key of the strong keyed hash spec.

[`class TableInitializerBase`](../../tf/contrib/lookup/TableInitializerBase.md): Base class for lookup table initializers.

[`class TextFileIdTableInitializer`](../../tf/contrib/lookup/TextFileIdTableInitializer.md): Table initializer for string to `int64` IDs tables from a text file.

[`class TextFileIndex`](../../tf/lookup/TextFileIndex.md): The key and value content to get from each line.

[`class TextFileInitializer`](../../tf/lookup/TextFileInitializer.md): Table initializers from a text file.

[`class TextFileStringTableInitializer`](../../tf/contrib/lookup/TextFileStringTableInitializer.md): Table initializer for `int64` IDs to string tables from a text file.

## Functions

[`index_table_from_file(...)`](../../tf/contrib/lookup/index_table_from_file.md): Returns a lookup table that converts a string tensor into int64 IDs.

[`index_table_from_tensor(...)`](../../tf/contrib/lookup/index_table_from_tensor.md): Returns a lookup table that converts a string tensor into int64 IDs.

[`index_to_string(...)`](../../tf/contrib/lookup/index_to_string.md): Maps `tensor` of indices into string values based on `mapping`. (deprecated)

[`index_to_string_table_from_file(...)`](../../tf/contrib/lookup/index_to_string_table_from_file.md): Returns a lookup table that maps a `Tensor` of indices into strings.

[`index_to_string_table_from_tensor(...)`](../../tf/contrib/lookup/index_to_string_table_from_tensor.md): Returns a lookup table that maps a `Tensor` of indices into strings.

[`string_to_index(...)`](../../tf/contrib/lookup/string_to_index.md): Maps `tensor` of strings into `int64` indices based on `mapping`. (deprecated)

[`string_to_index_table_from_file(...)`](../../tf/contrib/lookup/string_to_index_table_from_file.md): DEPRECATED FUNCTION

[`string_to_index_table_from_tensor(...)`](../../tf/contrib/lookup/string_to_index_table_from_tensor.md): DEPRECATED FUNCTION

## Other Members

* `FastHashSpec` <a id="FastHashSpec"></a>
