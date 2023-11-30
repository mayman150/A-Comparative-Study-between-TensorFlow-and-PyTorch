description: Type specification for a <a href="../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.RaggedTensorSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="experimental_as_proto"/>
<meta itemprop="property" content="experimental_from_proto"/>
<meta itemprop="property" content="experimental_type_proto"/>
<meta itemprop="property" content="from_value"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="is_subtype_of"/>
<meta itemprop="property" content="most_specific_common_supertype"/>
<meta itemprop="property" content="most_specific_compatible_type"/>
</div>

# tf.RaggedTensorSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_tensor.py">View source</a>



Type specification for a <a href="../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>.

Inherits From: [`TypeSpec`](../tf/TypeSpec.md), [`TraceType`](../tf/types/experimental/TraceType.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.RaggedTensorSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.RaggedTensorSpec(
    shape=None,
    dtype=<a href="../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    ragged_rank=None,
    row_splits_dtype=<a href="../tf/dtypes.md#int64"><code>tf.dtypes.int64</code></a>,
    flat_values_spec=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
The shape of the RaggedTensor, or `None` to allow any shape.  If a
shape is specified, then all ragged dimensions must have size `None`.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
<a href="../tf/dtypes/DType.md"><code>tf.DType</code></a> of values in the RaggedTensor.
</td>
</tr><tr>
<td>
`ragged_rank`<a id="ragged_rank"></a>
</td>
<td>
Python integer, the number of times the RaggedTensor's
flat_values is partitioned.  Defaults to `shape.ndims - 1`.
</td>
</tr><tr>
<td>
`row_splits_dtype`<a id="row_splits_dtype"></a>
</td>
<td>
`dtype` for the RaggedTensor's `row_splits` tensor. One
of <a href="../tf.md#int32"><code>tf.int32</code></a> or <a href="../tf.md#int64"><code>tf.int64</code></a>.
</td>
</tr><tr>
<td>
`flat_values_spec`<a id="flat_values_spec"></a>
</td>
<td>
TypeSpec for flat_value of the RaggedTensor. It shall be
provided when the flat_values is a CompositeTensor rather then Tensor.
If both `dtype` and `flat_values_spec` and  are provided, `dtype` must
be the same as `flat_values_spec.dtype`. (experimental)
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The <a href="../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a> specified by this type for the RaggedTensor.


```
>>> rt = tf.ragged.constant([["a"], ["b", "c"]], dtype=tf.string)
>>> tf.type_spec_from_value(rt).dtype
tf.string
```
</td>
</tr><tr>
<td>
`flat_values_spec`<a id="flat_values_spec"></a>
</td>
<td>
The `TypeSpec` of the flat_values of RaggedTensor.
</td>
</tr><tr>
<td>
`ragged_rank`<a id="ragged_rank"></a>
</td>
<td>
The number of times the RaggedTensor's flat_values is partitioned.

Defaults to `shape.ndims - 1`.

```
>>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
>>> tf.type_spec_from_value(values).ragged_rank
1
```

```
>>> rt1 = tf.RaggedTensor.from_uniform_row_length(values, 2)
>>> tf.type_spec_from_value(rt1).ragged_rank
2
```
</td>
</tr><tr>
<td>
`row_splits_dtype`<a id="row_splits_dtype"></a>
</td>
<td>
The <a href="../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a> of the RaggedTensor's `row_splits`.


```
>>> rt = tf.ragged.constant([[1, 2, 3], [4]], row_splits_dtype=tf.int64)
>>> tf.type_spec_from_value(rt).row_splits_dtype
tf.int64
```
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
The statically known shape of the RaggedTensor.


```
>>> rt = tf.ragged.constant([[0], [1, 2]])
>>> tf.type_spec_from_value(rt).shape
TensorShape([2, None])
```

```
>>> rt = tf.ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1)
>>> tf.type_spec_from_value(rt).shape
TensorShape([2, None, 2])
```
</td>
</tr><tr>
<td>
`value_type`<a id="value_type"></a>
</td>
<td>
The Python type for values that are compatible with this TypeSpec.

In particular, all values that are compatible with this TypeSpec must be an
instance of this type.
</td>
</tr>
</table>



## Methods

<h3 id="experimental_as_proto"><code>experimental_as_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_as_proto() -> struct_pb2.TypeSpecProto
</code></pre>

Returns a proto representation of the TypeSpec instance.

Do NOT override for custom non-TF types.

<h3 id="experimental_from_proto"><code>experimental_from_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>experimental_from_proto(
    proto: struct_pb2.TypeSpecProto
) -> 'TypeSpec'
</code></pre>

Returns a TypeSpec instance based on the serialized proto.

Do NOT override for custom non-TF types.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`proto`
</td>
<td>
Proto generated using 'experimental_as_proto'.
</td>
</tr>
</table>



<h3 id="experimental_type_proto"><code>experimental_type_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>experimental_type_proto() -> Type[struct_pb2.TypeSpecProto]
</code></pre>

Returns the type of proto associated with TypeSpec serialization.

Do NOT override for custom non-TF types.

<h3 id="from_value"><code>from_value</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_value(
    value
)
</code></pre>




<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    spec_or_value
)
</code></pre>

Returns true if `spec_or_value` is compatible with this TypeSpec.

Prefer using "is_subtype_of" and "most_specific_common_supertype" wherever
possible.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec_or_value`
</td>
<td>
A TypeSpec or TypeSpec associated value to compare against.
</td>
</tr>
</table>



<h3 id="is_subtype_of"><code>is_subtype_of</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_subtype_of(
    other: <a href="../tf/types/experimental/TraceType.md"><code>tf.types.experimental.TraceType</code></a>
) -> bool
</code></pre>

Returns True if `self` is a subtype of `other`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the
TypeSpec is covariant upon attributes that implement TraceType and
invariant upon rest of the attributes as well as the structure and type
of the TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A TraceType object.
</td>
</tr>
</table>



<h3 id="most_specific_common_supertype"><code>most_specific_common_supertype</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_common_supertype(
    others: Sequence[<a href="../tf/types/experimental/TraceType.md"><code>tf.types.experimental.TraceType</code></a>]
) -> Optional['TypeSpec']
</code></pre>

Returns the most specific supertype TypeSpec  of `self` and `others`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the
TypeSpec is covariant upon attributes that implement TraceType and
invariant upon rest of the attributes as well as the structure and type
of the TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`others`
</td>
<td>
A sequence of TraceTypes.
</td>
</tr>
</table>



<h3 id="most_specific_compatible_type"><code>most_specific_compatible_type</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_compatible_type(
    other: 'TypeSpec'
) -> 'TypeSpec'
</code></pre>

Returns the most specific TypeSpec compatible with `self` and `other`. (deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use most_specific_common_supertype instead.

Deprecated. Please use `most_specific_common_supertype` instead.
Do not override this function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A `TypeSpec`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If there is no TypeSpec that is compatible with both `self`
and `other`.
</td>
</tr>
</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
) -> bool
</code></pre>

Return self!=value.




