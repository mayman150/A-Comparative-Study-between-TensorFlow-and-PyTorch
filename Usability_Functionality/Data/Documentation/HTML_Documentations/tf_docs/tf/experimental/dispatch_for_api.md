description: Decorator that overrides the default implementation for a TensorFlow API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_api" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_api

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator that overrides the default implementation for a TensorFlow API.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_api`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_api(
    api, *signatures
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "dispatch target") will override the
default implementation for the API when the API is called with parameters that
match a specified type signature.  Signatures are specified using dictionaries
that map parameter names to type annotations.  E.g., in the following example,
`masked_add` will be called for <a href="../../tf/math/add.md"><code>tf.add</code></a> if both `x` and `y` are
`MaskedTensor`s:

```
>>> class MaskedTensor(tf.experimental.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
```

```
>>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor, 'y': MaskedTensor})
... def masked_add(x, y, name=None):
...   return MaskedTensor(x.values + y.values, x.mask & y.mask)
```

```
>>> mt = tf.add(MaskedTensor([1, 2], [True, False]), MaskedTensor(10, True))
>>> print(f"values={mt.values.numpy()}, mask={mt.mask.numpy()}")
values=[11 12], mask=[ True False]
```

If multiple type signatures are specified, then the dispatch target will be
called if any of the signatures match.  For example, the following code
registers `masked_add` to be called if `x` is a `MaskedTensor` *or* `y` is
a `MaskedTensor`.

```
>>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor}, {'y':MaskedTensor})
... def masked_add(x, y):
...   x_values = x.values if isinstance(x, MaskedTensor) else x
...   x_mask = x.mask if isinstance(x, MaskedTensor) else True
...   y_values = y.values if isinstance(y, MaskedTensor) else y
...   y_mask = y.mask if isinstance(y, MaskedTensor) else True
...   return MaskedTensor(x_values + y_values, x_mask & y_mask)
```

The type annotations in type signatures may be type objects (e.g.,
`MaskedTensor`), `typing.List` values, or `typing.Union` values.   For
example, the following will register `masked_concat` to be called if `values`
is a list of `MaskedTensor` values:

```
>>> @dispatch_for_api(tf.concat, {'values': typing.List[MaskedTensor]})
... def masked_concat(values, axis):
...   return MaskedTensor(tf.concat([v.values for v in values], axis),
...                       tf.concat([v.mask for v in values], axis))
```

Each type signature must contain at least one subclass of `tf.CompositeTensor`
(which includes subclasses of `tf.ExtensionType`), and dispatch will only be
triggered if at least one type-annotated parameter contains a
`CompositeTensor` value.  This rule avoids invoking dispatch in degenerate
cases, such as the following examples:

* `@dispatch_for_api(tf.concat, {'values': List[MaskedTensor]})`: Will not
  dispatch to the decorated dispatch target when the user calls
  `tf.concat([])`.

* `@dispatch_for_api(tf.add, {'x': Union[MaskedTensor, Tensor], 'y':
  Union[MaskedTensor, Tensor]})`: Will not dispatch to the decorated dispatch
  target when the user calls `tf.add(tf.constant(1), tf.constant(2))`.

The dispatch target's signature must match the signature of the API that is
being overridden.  In particular, parameters must have the same names, and
must occur in the same order.  The dispatch target may optionally elide the
"name" parameter, in which case it will be wrapped with a call to
<a href="../../tf/name_scope.md"><code>tf.name_scope</code></a> when appropraite.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`api`<a id="api"></a>
</td>
<td>
The TensorFlow API to override.
</td>
</tr><tr>
<td>
`*signatures`<a id="*signatures"></a>
</td>
<td>
Dictionaries mapping parameter names or indices to type
annotations, specifying when the dispatch target should be called.  In
particular, the dispatch target will be called if any signature matches;
and a signature matches if all of the specified parameters have types that
match with the indicated type annotations.  If no signatures are
specified, then a signature will be read from the dispatch target
function's type annotations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator that overrides the default implementation for `api`.
</td>
</tr>

</table>


#### Registered APIs

The TensorFlow APIs that may be overridden by `@dispatch_for_api` are:

<<API_LIST>>