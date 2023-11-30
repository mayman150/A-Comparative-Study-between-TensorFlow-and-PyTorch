description: Public API for tf._api.v2.experimental namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.experimental" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="float8_e4m3fn"/>
<meta itemprop="property" content="float8_e5m2"/>
<meta itemprop="property" content="int4"/>
<meta itemprop="property" content="uint4"/>
</div>

# Module: tf.compat.v1.experimental

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.experimental namespace



## Modules

[`extension_type`](../../../tf/compat/v1/experimental/extension_type.md) module: Public API for tf._api.v2.experimental.extension_type namespace

## Classes

[`class BatchableExtensionType`](../../../tf/experimental/BatchableExtensionType.md): An ExtensionType that can be batched and unbatched.

[`class DynamicRaggedShape`](../../../tf/experimental/DynamicRaggedShape.md): The shape of a ragged or dense tensor.

[`class ExtensionType`](../../../tf/experimental/ExtensionType.md): Base class for TensorFlow `ExtensionType` classes.

[`class ExtensionTypeBatchEncoder`](../../../tf/experimental/ExtensionTypeBatchEncoder.md): Class used to encode and decode extension type values for batching.

[`class ExtensionTypeSpec`](../../../tf/experimental/ExtensionTypeSpec.md): Base class for tf.ExtensionType TypeSpec.

[`class Optional`](../../../tf/experimental/Optional.md): Represents a value that may or may not be present.

[`class RowPartition`](../../../tf/experimental/RowPartition.md): Partitioning of a sequence of values into contiguous subsequences ("rows").

[`class StructuredTensor`](../../../tf/experimental/StructuredTensor.md): A multidimensional collection of structures with the same schema.

## Functions

[`async_clear_error(...)`](../../../tf/experimental/async_clear_error.md): Clear pending operations and error statuses in async execution.

[`async_scope(...)`](../../../tf/experimental/async_scope.md): Context manager for grouping async operations.

[`dispatch_for_api(...)`](../../../tf/experimental/dispatch_for_api.md): Decorator that overrides the default implementation for a TensorFlow API.

[`dispatch_for_binary_elementwise_apis(...)`](../../../tf/experimental/dispatch_for_binary_elementwise_apis.md): Decorator to override default implementation for binary elementwise APIs.

[`dispatch_for_binary_elementwise_assert_apis(...)`](../../../tf/experimental/dispatch_for_binary_elementwise_assert_apis.md): Decorator to override default implementation for binary elementwise assert APIs.

[`dispatch_for_unary_elementwise_apis(...)`](../../../tf/experimental/dispatch_for_unary_elementwise_apis.md): Decorator to override default implementation for unary elementwise APIs.

[`enable_strict_mode(...)`](../../../tf/experimental/enable_strict_mode.md): If called, enables strict mode for all behaviors.

[`function_executor_type(...)`](../../../tf/experimental/function_executor_type.md): Context manager for setting the executor of eager defined functions.

[`output_all_intermediates(...)`](../../../tf/compat/v1/experimental/output_all_intermediates.md): Whether to output all intermediates from functional control flow ops.

[`register_filesystem_plugin(...)`](../../../tf/experimental/register_filesystem_plugin.md): Loads a TensorFlow FileSystem plugin.

[`unregister_dispatch_for(...)`](../../../tf/experimental/unregister_dispatch_for.md): Unregisters a function that was registered with `@dispatch_for_*`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
float8_e4m3fn<a id="float8_e4m3fn"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>


8-bit float with 4 exponent bits and 3 mantissa bits, with extended finite range.  This type has no representation for inf, and only two NaN values: 0xFF for negative NaN, and 0x7F for positive NaN.
</td>
</tr><tr>
<td>
float8_e5m2<a id="float8_e5m2"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>


8-bit float with 5 exponent bits and 2 mantissa bits.
</td>
</tr><tr>
<td>
int4<a id="int4"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>


Signed 4-bit integer.
</td>
</tr><tr>
<td>
uint4<a id="uint4"></a>
</td>
<td>
Instance of <a href="../../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>


Unsigned 4-bit integer.
</td>
</tr>
</table>

