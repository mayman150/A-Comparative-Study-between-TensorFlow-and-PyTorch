description: Describes the type of a tf.Tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.TensorSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="experimental_as_proto"/>
<meta itemprop="property" content="experimental_from_proto"/>
<meta itemprop="property" content="experimental_type_proto"/>
<meta itemprop="property" content="from_spec"/>
<meta itemprop="property" content="from_tensor"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="is_subtype_of"/>
<meta itemprop="property" content="most_specific_common_supertype"/>
<meta itemprop="property" content="most_specific_compatible_type"/>
</div>

# tf.TensorSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>



Describes the type of a tf.Tensor.

Inherits From: [`TypeSpec`](../tf/TypeSpec.md), [`TraceType`](../tf/types/experimental/TraceType.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.TensorSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.TensorSpec(
    shape,
    dtype=<a href="../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

```
>>> t = tf.constant([[1,2,3],[4,5,6]])
>>> tf.TensorSpec.from_tensor(t)
TensorSpec(shape=(2, 3), dtype=tf.int32, name=None)
```

Contains metadata for describing the nature of <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> objects
accepted or returned by some TensorFlow APIs.

For example, it can be used to constrain the type of inputs accepted by
a tf.function:

```
>>> @tf.function(input_signature=[tf.TensorSpec([1, None])])
... def constrained_foo(t):
...   print("tracing...")
...   return t
```

Now the <a href="../tf/function.md"><code>tf.function</code></a> is able to assume that `t` is always of the type
`tf.TensorSpec([1, None])` which will avoid retracing as well as enforce the
type restriction on inputs.

As a result, the following call with tensor of type `tf.TensorSpec([1, 2])`
triggers a trace and succeeds:
```
>>> constrained_foo(tf.constant([[1., 2]])).numpy()
tracing...
array([[1., 2.]], dtype=float32)
```

The following subsequent call with tensor of type `tf.TensorSpec([1, 4])`
does not trigger a trace and succeeds:
```
>>> constrained_foo(tf.constant([[1., 2, 3, 4]])).numpy()
array([[1., 2., 3., 4.], dtype=float32)
```

But the following call with tensor of type `tf.TensorSpec([2, 2])` fails:
```
>>> constrained_foo(tf.constant([[1., 2], [3, 4]])).numpy()
Traceback (most recent call last):
...
TypeError: Binding inputs to tf.function `constrained_foo` failed ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
Value convertible to <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a>. The shape of the tensor.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Value convertible to <a href="../tf/dtypes/DType.md"><code>tf.DType</code></a>. The type of the tensor values.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the Tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
If shape is not convertible to a <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a>, or dtype is
not convertible to a <a href="../tf/dtypes/DType.md"><code>tf.DType</code></a>.
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
Returns the `dtype` of elements in the tensor.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Returns the (optionally provided) name of the described tensor.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
Returns the `TensorShape` that represents the shape of the tensor.
</td>
</tr><tr>
<td>
`value_type`<a id="value_type"></a>
</td>
<td>
The Python type for values that are compatible with this TypeSpec.
</td>
</tr>
</table>



## Methods

<h3 id="experimental_as_proto"><code>experimental_as_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_as_proto() -> struct_pb2.TensorSpecProto
</code></pre>

Returns a proto representation of the TensorSpec instance.


<h3 id="experimental_from_proto"><code>experimental_from_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>experimental_from_proto(
    proto: struct_pb2.TensorSpecProto
) -> 'TensorSpec'
</code></pre>

Returns a TensorSpec instance based on the serialized proto.


<h3 id="experimental_type_proto"><code>experimental_type_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>experimental_type_proto() -> Type[struct_pb2.TensorSpecProto]
</code></pre>

Returns the type of proto associated with TensorSpec serialization.


<h3 id="from_spec"><code>from_spec</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_spec(
    spec, name=None
)
</code></pre>

Returns a `TensorSpec` with the same shape and dtype as `spec`.

```
>>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="OriginalName")
>>> tf.TensorSpec.from_spec(spec, "NewName")
TensorSpec(shape=(8, 3), dtype=tf.int32, name='NewName')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The `TypeSpec` used to create the new `TensorSpec`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name for the new `TensorSpec`.  Defaults to `spec.name`.
</td>
</tr>
</table>



<h3 id="from_tensor"><code>from_tensor</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_tensor(
    tensor, name=None
)
</code></pre>

Returns a `TensorSpec` that describes `tensor`.

```
>>> tf.TensorSpec.from_tensor(tf.constant([1, 2, 3]))
TensorSpec(shape=(3,), dtype=tf.int32, name=None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor`
</td>
<td>
The <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> that should be described.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the `TensorSpec`.  Defaults to `tensor.op.name`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TensorSpec` that describes `tensor`.
</td>
</tr>

</table>



<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    spec_or_tensor
)
</code></pre>

Returns True if spec_or_tensor is compatible with this TensorSpec.

Two tensors are considered compatible if they have the same dtype
and their shapes are compatible (see <a href="../tf/TensorShape.md#is_compatible_with"><code>tf.TensorShape.is_compatible_with</code></a>).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec_or_tensor`
</td>
<td>
A tf.TensorSpec or a tf.Tensor
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if spec_or_tensor is compatible with self.
</td>
</tr>

</table>



<h3 id="is_subtype_of"><code>is_subtype_of</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_subtype_of(
    other
)
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

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




