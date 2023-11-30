description: Wrapper for <a href="../tf/Graph.md#control_dependencies"><code>Graph.control_dependencies()</code></a> using the default graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.control_dependencies" />
<meta itemprop="path" content="Stable" />
</div>

# tf.control_dependencies

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Wrapper for <a href="../tf/Graph.md#control_dependencies"><code>Graph.control_dependencies()</code></a> using the default graph.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.control_dependencies`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.control_dependencies(
    control_inputs
)
</code></pre>



<!-- Placeholder for "Used in" -->

See <a href="../tf/Graph.md#control_dependencies"><code>tf.Graph.control_dependencies</code></a> for more details.

In TensorFlow 2 with eager and/or Autograph, you should not need this method
most of the times, as ops execute in the expected order thanks to automatic
control dependencies. Only use it to manually control ordering, for example as
a workaround to known issues such as <a href="../tf/function.md"><code>tf.function</code></a> with `tf.debugging.assert*`
and <a href="../tf/py_function.md"><code>tf.py_function</code></a>.
For example:

```
>>> @tf.function(
...   input_signature=[tf.TensorSpec([None, None], tf.float32),
...                    tf.TensorSpec([None, None], tf.float32)])
... def my_assert_func_1(x, bias):
...   # `tf.function` attempts to execute `tf.math.add` in parallel to
...   # `assert_equal`. As a result an error can get raised from `tf.math.add`
...   # without triggering the assertion error.
...   tf.assert_equal(tf.shape(x)[1],
...                   tf.shape(bias)[1],
...                   message='bad shape')
...   return x + bias
```

```
>>> # Error raised in either `add` or `assert`
>>> my_assert_func_1(tf.ones((2, 5)), tf.ones((2, 7)))
Traceback (most recent call last):
   ...
InvalidArgumentError: ...
```


```
>>> @tf.function(
...   input_signature=[tf.TensorSpec([None, None], tf.float32),
...                    tf.TensorSpec([None, None], tf.float32)])
... def my_assert_func_2(x, bias):
...   with tf.control_dependencies(
...       [tf.assert_equal(tf.shape(x)[1],
...                       tf.shape(bias)[1],
...                       message='bad shape')]):
...     return x + bias
```

```
>>> # Error raised in `assert`
>>> my_assert_func_2(tf.ones((2, 5)), tf.ones((2, 7)))
Traceback (most recent call last):
   ...
InvalidArgumentError: ...
```

When eager execution is enabled, any callable object in the `control_inputs`
list will be called.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`control_inputs`<a id="control_inputs"></a>
</td>
<td>
A list of `Operation` or `Tensor` objects which must be
executed or computed before running the operations defined in the context.
Can also be `None` to clear the control dependencies. If eager execution
is enabled, any callable object in the `control_inputs` list will be
called.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A context manager that specifies control dependencies for all
operations constructed within the context.
</td>
</tr>

</table>

