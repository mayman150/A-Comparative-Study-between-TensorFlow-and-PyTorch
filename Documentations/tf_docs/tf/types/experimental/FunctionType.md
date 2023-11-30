description: Represents the type of a TensorFlow callable.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.FunctionType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="empty"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bind"/>
<meta itemprop="property" content="bind_partial"/>
<meta itemprop="property" content="from_builtin"/>
<meta itemprop="property" content="from_callable"/>
<meta itemprop="property" content="from_function"/>
<meta itemprop="property" content="replace"/>
</div>

# tf.types.experimental.FunctionType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>



Represents the type of a TensorFlow callable.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.types.experimental.FunctionType`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.types.experimental.FunctionType(
    parameters=None, *, return_annotation, __validate_parameters__=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

FunctionType inherits from inspect.Signature which canonically represents the
structure (and optionally type) information of input parameters and output of
a Python function. Additionally, it integrates with the tf.function type
system (<a href="../../../tf/types/experimental/TraceType.md"><code>tf.types.experimental.TraceType</code></a>) to provide a holistic
representation of the the I/O contract of the callable. It is used for:
  - Canonicalization and type-checking of Python input arguments
  - Type-based dispatch to concrete functions
  - Packing/unpacking structured python values to Tensors
  - Generation of structured placeholder values for tracing



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`parameters`<a id="parameters"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`return_annotation`<a id="return_annotation"></a>
</td>
<td>

</td>
</tr>
</table>



## Child Classes
[`class empty`](../../../tf/types/experimental/FunctionType/empty.md)

## Methods

<h3 id="bind"><code>bind</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bind(
    *args, **kwargs
)
</code></pre>

Get a BoundArguments object, that maps the passed `args` and `kwargs` to the function's signature.
  Raises `TypeError`
if the passed arguments can not be bound.

<h3 id="bind_partial"><code>bind_partial</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bind_partial(
    *args, **kwargs
)
</code></pre>

Get a BoundArguments object, that partially maps the passed `args` and `kwargs` to the function's signature.
Raises `TypeError` if the passed arguments can not be bound.

<h3 id="from_builtin"><code>from_builtin</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_builtin(
    func
)
</code></pre>

Constructs Signature for the given builtin function.

Deprecated since Python 3.5, use `Signature.from_callable()`.

<h3 id="from_callable"><code>from_callable</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_callable(
    obj, *, follow_wrapped=True
)
</code></pre>

Constructs Signature for the given callable object.


<h3 id="from_function"><code>from_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_function(
    func
)
</code></pre>

Constructs Signature for the given python function.

Deprecated since Python 3.5, use `Signature.from_callable()`.

<h3 id="replace"><code>replace</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace(
    *, parameters=_void, return_annotation=_void
)
</code></pre>

Creates a customized copy of the Signature.
Pass 'parameters' and/or 'return_annotation' arguments
to override them in the new copy.

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




