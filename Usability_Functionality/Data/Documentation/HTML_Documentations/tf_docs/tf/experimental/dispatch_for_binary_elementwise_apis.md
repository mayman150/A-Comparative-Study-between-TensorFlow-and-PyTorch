description: Decorator to override default implementation for binary elementwise APIs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_binary_elementwise_apis" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_binary_elementwise_apis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator to override default implementation for binary elementwise APIs.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_binary_elementwise_apis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_binary_elementwise_apis(
    x_type, y_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "elementwise api handler") overrides
the default implementation for any binary elementwise API whenever the value
for the first two arguments (typically named `x` and `y`) match the specified
type annotations.  The elementwise api handler is called with two arguments:

  `elementwise_api_handler(api_func, x, y)`

Where `x` and `y` are the first two arguments to the elementwise api, and
`api_func` is a TensorFlow function that takes two parameters and performs the
elementwise operation (e.g., <a href="../../tf/math/add.md"><code>tf.add</code></a>).

The following example shows how this decorator can be used to update all
binary elementwise operations to handle a `MaskedTensor` type:

```
>>> class MaskedTensor(tf.experimental.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
>>> @dispatch_for_binary_elementwise_apis(MaskedTensor, MaskedTensor)
... def binary_elementwise_api_handler(api_func, x, y):
...   return MaskedTensor(api_func(x.values, y.values), x.mask & y.mask)
>>> a = MaskedTensor([1, 2, 3, 4, 5], [True, True, True, True, False])
>>> b = MaskedTensor([2, 4, 6, 8, 0], [True, True, True, False, True])
>>> c = tf.add(a, b)
>>> print(f"values={c.values.numpy()}, mask={c.mask.numpy()}")
values=[ 3 6 9 12 5], mask=[ True True True False False]
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
</td>
</tr><tr>
<td>
`y_type`<a id="y_type"></a>
</td>
<td>
A type annotation indicating when the api handler should be called.
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

The binary elementwise APIs are:

<<API_LIST>>