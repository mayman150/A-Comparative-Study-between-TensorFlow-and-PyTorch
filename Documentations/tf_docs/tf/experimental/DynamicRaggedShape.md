description: The shape of a ragged or dense tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.DynamicRaggedShape" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="Spec"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="from_lengths"/>
<meta itemprop="property" content="from_row_partitions"/>
<meta itemprop="property" content="from_tensor"/>
<meta itemprop="property" content="is_uniform"/>
<meta itemprop="property" content="static_lengths"/>
<meta itemprop="property" content="with_dtype"/>
</div>

# tf.experimental.DynamicRaggedShape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>



The shape of a ragged or dense tensor.

Inherits From: [`BatchableExtensionType`](../../tf/experimental/BatchableExtensionType.md), [`ExtensionType`](../../tf/experimental/ExtensionType.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.DynamicRaggedShape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.DynamicRaggedShape(
    row_partitions: Sequence[<a href="../../tf/experimental/RowPartition.md"><code>tf.experimental.RowPartition</code></a>],
    inner_shape: <a href="../../tf/types/experimental/TensorLike.md"><code>tf.types.experimental.TensorLike</code></a>,
    dtype: Optional[<a href="../../tf/dtypes/DType.md"><code>tf.dtypes.DType</code></a>] = None,
    validate: bool = False,
    static_inner_shape: ... = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Ragged shapes are encoded using two fields:

* `inner_shape`: An integer vector giving the shape of a dense tensor.
* `row_partitions`: A list of `RowPartition` objects, describing how
  that flat shape should be partitioned to add ragged axes.

If a DynamicRaggedShape is the shape of a RaggedTensor rt, then:
1. row_partitions = rt._nested_row_partitions
   (and thus len(row_partitions) > 0)
2. inner_shape is the shape of rt.flat_values

If a DynamicRaggedShape is the shape of a dense tensor t, then:
1. row_partitions = []
2. inner_shape is the shape of t.

#### Examples:



The following table gives a few examples (where `RP(lengths)` is short
for `RowPartition.from_lengths(lengths)`):

Row Partitions              | Inner Shape  | Example Tensor
--------------------------- | ------------ | ----------------------------
[]                          | [2, 3]       | `[[1, 2, 3], [4, 5, 6]]`
[RP([2, 0, 3])]             | [5]          | `[[1, 2], [], [3, 4, 5]]`
[RP([2, 1])]                | [3, 2]       | `[[[1, 2], [3, 4]], [[5, 6]]]`
[RP([2, 1]), RP([2, 1, 2])] | [5]          | `[[[1, 2], [3]], [[4, 5]]]`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`row_partitions`<a id="row_partitions"></a>
</td>
<td>
the row_partitions of the shape.
</td>
</tr><tr>
<td>
`inner_shape`<a id="inner_shape"></a>
</td>
<td>
if len(row_partitions) > 0, the shape of the flat_values.
Otherwise, the shape of the tensor.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
tf.int64, tf.int32, or None representing the preferred dtype.
</td>
</tr><tr>
<td>
`validate`<a id="validate"></a>
</td>
<td>
if true, dynamic validation is applied to the shape.
</td>
</tr><tr>
<td>
`static_inner_shape`<a id="static_inner_shape"></a>
</td>
<td>
if len(row_partitions) > 0, the static shape of the
flat_values. Otherwise, the static shape of the tensor. Should be
convertible to a TensorShape.
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
The dtype of the shape -- one of tf.int32 or tf.int64.
</td>
</tr><tr>
<td>
`inner_rank`<a id="inner_rank"></a>
</td>
<td>
The rank of inner_shape.
</td>
</tr><tr>
<td>
`inner_shape`<a id="inner_shape"></a>
</td>
<td>
The inner dimension sizes for this shape.
</td>
</tr><tr>
<td>
`num_row_partitions`<a id="num_row_partitions"></a>
</td>
<td>
The number of row_partitions of the shape.
</td>
</tr><tr>
<td>
`rank`<a id="rank"></a>
</td>
<td>
The number of dimensions in this shape, or None if unknown.
</td>
</tr><tr>
<td>
`row_partitions`<a id="row_partitions"></a>
</td>
<td>
The row_partitions of the shape.
</td>
</tr>
</table>



## Child Classes
[`class Spec`](../../tf/experimental/DynamicRaggedShape/Spec.md)

## Methods

<h3 id="from_lengths"><code>from_lengths</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_lengths(
    lengths: Sequence[Union[Sequence[int], int]],
    num_row_partitions=None,
    dtype=<a href="../../tf/dtypes.md#int64"><code>tf.dtypes.int64</code></a>
)
</code></pre>

Creates a shape with the given lengths and num_row_partitions.

The lengths can either be a nonnegative int or a list of nonnegative ints.

If num_row_partitions is None, then the minimal num_row_partitions is used.

For example, [2, (3, 2)] is the shape of [[0, 0, 0], [0, 0]], and
[2, 2] is the shape of [[0, 0], [0, 0]]

This chooses the minimal num_row_partitions required (including zero).

The following table gives a few examples (where `RP(lengths)` is short
for `RowPartition.from_lengths(lengths)`):

#### For example:


from_lengths           | row_partitions            | inner_shape
---------------------- | --------------------------| -------------
[]                     | []                        | []
[2, (3, 2)]            | [RP([3, 2])]              | [5]
[2, 2]                 | []                        | [2, 2]
[2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
[2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
[2, 2, 3]              | []                        | [2, 2, 3]
[2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]

If we want the row partitions to end with uniform row partitions, then
we can set num_row_partitions.

For example,
below URP(3, 12) is RowPartition.from_uniform_row_length(3, 12)

from_lengths   | num_row_partitions | row_partitions           | inner_shape
---------------| -------------------|--------------------------|------------
[2, (3, 2), 2] | 2                  | [RP([3, 2]), URP(2, 10)] | [10]
[2, 2]         | 1                  | [URP(2, 4)]              | [4]
[2, 2, 3]      | 0                  | []                       | [2, 2, 3]
[2, 2, 3]      | 1                  | [URP(2, 4)]              | [4, 3]
[2, 2, 3]      | 2                  | [URP(2, 4), URP(3, 12)]  | [12]



Representing the shapes from init():

from_lengths             | Tensor Example
------------------------ | ------------------------------
`[2, 3]`                 | `[[1, 2, 3], [4, 5, 6]]`
`[3, (2, 0, 3)]`         | `[[1, 2], [], [3, 4, 5]]`
`[2, (2, 1), 2]`         | `[[[1, 2], [3, 4]], [[5, 6]]]`
`[2, (2, 1), (2, 1, 2)]` | `[[[1, 2], [3]], [[4, 5]]]`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`lengths`
</td>
<td>
the lengths of sublists along each axis.
</td>
</tr><tr>
<td>
`num_row_partitions`
</td>
<td>
the num_row_partitions of the result or None
indicating the minimum number of row_partitions.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the dtype of the shape (tf.int32 or tf.int64).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a new DynamicRaggedShape
</td>
</tr>

</table>



<h3 id="from_row_partitions"><code>from_row_partitions</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_row_partitions(
    row_partitions, dtype=None
)
</code></pre>

Create a shape from row_partitions.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`row_partitions`
</td>
<td>
a nonempty list of RowPartition objects.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the dtype to use, or None to use the row_partitions dtype.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a DynamicRaggedShape with inner_rank==1.
</td>
</tr>

</table>



<h3 id="from_tensor"><code>from_tensor</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_tensor(
    t, dtype=None
)
</code></pre>

Constructs a ragged shape for a potentially ragged tensor.


<h3 id="is_uniform"><code>is_uniform</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_uniform(
    axis
)
</code></pre>

Returns true if the indicated dimension is uniform.


<h3 id="static_lengths"><code>static_lengths</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>static_lengths(
    ragged_lengths=True
)
</code></pre>

Returns a list of statically known axis lengths.

This represents what values are known. For each row partition, it presents
either the uniform row length (if statically known),
the list of row lengths, or none if it is not statically known.
For the inner shape, if the rank is known, then each dimension is reported
if known, and None otherwise. If the rank of the inner shape is not known,
then the returned list ends with an ellipsis.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`ragged_lengths`
</td>
<td>
If false, returns None for all ragged dimensions.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Sequence[Union[Sequence[int],int, None]] of lengths, with a possible
Ellipsis at the end.
</td>
</tr>

</table>



<h3 id="with_dtype"><code>with_dtype</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_dtype(
    dtype
)
</code></pre>

Change the dtype of the shape.


<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/dynamic_ragged_shape.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    index
)
</code></pre>

Returns a dimension or a slice of the shape.

Ragged shapes can have ragged dimensions that depend upon other dimensions.
Therefore, if you ask for a dimension that is ragged, this function returns
a ValueError. For similar reasons, if a slice is selected that includes
a ragged dimension without including the zero dimension, then this fails.

Any slice that does not start at zero will return a shape
with num_row_partitions == 0.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`index`
</td>
<td>
the index: can be an int or a slice.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`IndexError`
</td>
<td>
if the index is not in range.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if the rank is unknown, or a ragged rank is requested
incorrectly.
</td>
</tr>
</table>



<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




