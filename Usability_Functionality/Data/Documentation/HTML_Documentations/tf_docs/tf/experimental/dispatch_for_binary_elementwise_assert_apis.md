description: Decorator to override default implementation for binary elementwise assert APIs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_binary_elementwise_assert_apis" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_binary_elementwise_assert_apis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator to override default implementation for binary elementwise assert APIs.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_binary_elementwise_assert_apis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_binary_elementwise_assert_apis(
    x_type, y_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "elementwise assert handler")
overrides the default implementation for any binary elementwise assert API
whenever the value for the first two arguments (typically named `x` and `y`)
match the specified type annotations.  The handler is called with two
arguments:

  `elementwise_assert_handler(assert_func, x, y)`

Where `x` and `y` are the first two arguments to the binary elementwise assert
operation, and `assert_func` is a TensorFlow function that takes two
parameters and performs the elementwise assert operation (e.g.,
<a href="../../tf/debugging/assert_equal.md"><code>tf.debugging.assert_equal</code></a>).

The following example shows how this decorator can be used to update all
binary elementwise assert operations to handle a `MaskedTensor` type:

```
>>> class MaskedTensor(tf.experimental.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
>>> @dispatch_for_binary_elementwise_assert_apis(MaskedTensor, MaskedTensor)
... def binary_elementwise_assert_api_handler(assert_func, x, y):
...   merged_mask = tf.logical_and(x.mask, y.mask)
...   selected_x_values = tf.boolean_mask(x.values, merged_mask)
...   selected_y_values = tf.boolean_mask(y.values, merged_mask)
...   assert_func(selected_x_values, selected_y_values)
>>> a = MaskedTensor([1, 1, 0, 1, 1], [False, False, True, True, True])
>>> b = MaskedTensor([2, 2, 0, 2, 2], [True, True, True, False, False])
>>> tf.debugging.assert_equal(a, b) # assert passed; no exception was thrown
```

```
>>> a = MaskedTensor([1, 1, 1, 1, 1], [True, True, True, True, True])
>>> b = MaskedTensor([0, 0, 0, 0, 2], [True, True, True, True, True])
>>> tf.debugging.assert_greater(a, b)
Traceback (most recent call last):
...
InvalidArgumentError: Condition x > y did not hold.
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

The binary elementwise assert APIs are:

<<API_LIST>>