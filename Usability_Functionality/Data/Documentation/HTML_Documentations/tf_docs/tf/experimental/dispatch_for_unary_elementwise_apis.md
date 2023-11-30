description: Decorator to override default implementation for unary elementwise APIs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_unary_elementwise_apis" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_unary_elementwise_apis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator to override default implementation for unary elementwise APIs.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_unary_elementwise_apis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_unary_elementwise_apis(
    x_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "elementwise api handler") overrides
the default implementation for any unary elementwise API whenever the value
for the first argument (typically named `x`) matches the type annotation
`x_type`. The elementwise api handler is called with two arguments:

  `elementwise_api_handler(api_func, x)`

Where `api_func` is a function that takes a single parameter and performs the
elementwise operation (e.g., <a href="../../tf/math/abs.md"><code>tf.abs</code></a>), and `x` is the first argument to the
elementwise api.

The following example shows how this decorator can be used to update all
unary elementwise operations to handle a `MaskedTensor` type:

```
>>> class MaskedTensor(tf.experimental.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
>>> @dispatch_for_unary_elementwise_apis(MaskedTensor)
... def unary_elementwise_api_handler(api_func, x):
...   return MaskedTensor(api_func(x.values), x.mask)
>>> mt = MaskedTensor([1, -2, -3], [True, False, True])
>>> abs_mt = tf.abs(mt)
>>> print(f"values={abs_mt.values.numpy()}, mask={abs_mt.mask.numpy()}")
values=[1 2 3], mask=[ True False True]
```

For unary elementwise operations that take extra arguments beyond `x`, those
arguments are *not* passed to the elementwise api handler, but are
automatically added when `api_func` is called.  E.g., in the following
example, the `dtype` parameter is not passed to
`unary_elementwise_api_handler`, but is added by `api_func`.

```
>>> ones_mt = tf.ones_like(mt, dtype=tf.float32)
>>> print(f"values={ones_mt.values.numpy()}, mask={ones_mt.mask.numpy()}")
values=[1.0 1.0 1.0], mask=[ True False True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x_type`<a id="x_type"></a>
</td>
<td>
A type annotation indicating when the api handler should be called.
See `dispatch_for_api` for a list of supported annotation types.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator.
</td>
</tr>

</table>


#### Registered APIs

The unary elementwise APIs are:

<<API_LIST>>