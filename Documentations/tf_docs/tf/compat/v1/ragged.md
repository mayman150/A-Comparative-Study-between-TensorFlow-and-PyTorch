description: Public API for tf._api.v2.ragged namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.ragged" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.ragged

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.ragged namespace



## Classes

[`class RaggedTensorValue`](../../../tf/compat/v1/ragged/RaggedTensorValue.md): Represents the value of a `RaggedTensor`.

## Functions

[`boolean_mask(...)`](../../../tf/ragged/boolean_mask.md): Applies a boolean mask to `data` without flattening the mask dimensions.

[`constant(...)`](../../../tf/ragged/constant.md): Constructs a constant RaggedTensor from a nested Python list.

[`constant_value(...)`](../../../tf/compat/v1/ragged/constant_value.md): Constructs a RaggedTensorValue from a nested Python list.

[`cross(...)`](../../../tf/ragged/cross.md): Generates feature cross from a list of tensors.

[`cross_hashed(...)`](../../../tf/ragged/cross_hashed.md): Generates hashed feature cross from a list of tensors.

[`map_flat_values(...)`](../../../tf/ragged/map_flat_values.md): Applies `op` to the `flat_values` of one or more RaggedTensors.

[`placeholder(...)`](../../../tf/compat/v1/ragged/placeholder.md): Creates a placeholder for a <a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> that will always be fed.

[`range(...)`](../../../tf/ragged/range.md): Returns a `RaggedTensor` containing the specified sequences of numbers.

[`row_splits_to_segment_ids(...)`](../../../tf/ragged/row_splits_to_segment_ids.md): Generates the segmentation corresponding to a RaggedTensor `row_splits`.

[`segment_ids_to_row_splits(...)`](../../../tf/ragged/segment_ids_to_row_splits.md): Generates the RaggedTensor `row_splits` corresponding to a segmentation.

[`stack(...)`](../../../tf/ragged/stack.md): Stacks a list of rank-`R` tensors into one rank-`(R+1)` `RaggedTensor`.

[`stack_dynamic_partitions(...)`](../../../tf/ragged/stack_dynamic_partitions.md): Stacks dynamic partitions of a Tensor or RaggedTensor.

