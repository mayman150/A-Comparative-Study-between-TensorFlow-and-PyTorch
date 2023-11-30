description: Wraps a python function into a TensorFlow op that executes it eagerly.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.py_function" />
<meta itemprop="path" content="Stable" />
</div>

# tf.py_function

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/script_ops.py">View source</a>



Wraps a python function into a TensorFlow op that executes it eagerly.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.py_function`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.py_function(
    func=None, inp=None, Tout=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Using <a href="../tf/py_function.md"><code>tf.py_function</code></a> inside a <a href="../tf/function.md"><code>tf.function</code></a> allows you to run a python
function using eager execution, inside the <a href="../tf/function.md"><code>tf.function</code></a>'s graph.
This has two main affects:

1. This allows you to use nofunc=None, inp=None, Tout=Nonen tensorflow code
inside your <a href="../tf/function.md"><code>tf.function</code></a>.
2. It allows you to run python control logic in a <a href="../tf/function.md"><code>tf.function</code></a> without
relying on <a href="../tf/autograph.md"><code>tf.autograph</code></a> to convert the code to use tensorflow control logic
(tf.cond, tf.while_loop).

Both of these features can be useful for debgging.

Since <a href="../tf/py_function.md"><code>tf.py_function</code></a> operates on `Tensor`s it is still
differentiable (once).

There are two ways to use this function:

### As a decorator

Use <a href="../tf/py_function.md"><code>tf.py_function</code></a> as a decorator to ensure the function always runs
eagerly.

When using <a href="../tf/py_function.md"><code>tf.py_function</code></a> as a decorator:

* you must set `Tout`
* you may set `name`
* you must not set `func` or `inp`

For example, you might use <a href="../tf/py_function.md"><code>tf.py_function</code></a> to
implement the log huber function.

```
>>> @tf.py_function(Tout=tf.float32)
... def py_log_huber(x, m):
...   print('Running with eager execution.')
...   if tf.abs(x) <= m:
...     return x**2
...   else:
...     return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))
```

Under eager execution the function operates normally:

```
>>> x = tf.constant(1.0)
>>> m = tf.constant(2.0)
>>>
>>> print(py_log_huber(x,m).numpy())
Running with eager execution.
1.0
```

Inside a <a href="../tf/function.md"><code>tf.function</code></a> the <a href="../tf/py_function.md"><code>tf.py_function</code></a> is not converted to a <a href="../tf/Graph.md"><code>tf.Graph</code></a>.:

```
>>> @tf.function
... def tf_wrapper(x):
...   print('Tracing.')
...   m = tf.constant(2.0)
...   return py_log_huber(x,m)
```

The <a href="../tf/py_function.md"><code>tf.py_function</code></a> only executes eagerly, and only when the <a href="../tf/function.md"><code>tf.function</code></a>
is called:

```
>>> print(tf_wrapper(x).numpy())
Tracing.
Running with eager execution.
1.0
>>> print(tf_wrapper(x).numpy())
Running with eager execution.
1.0
```


Gradients work as exeppcted:

```
>>> with tf.GradientTape() as t:
...   t.watch(x)
...   y = tf_wrapper(x)
Running with eager execution.
>>>
>>> t.gradient(y, x).numpy()
2.0
```

### Inplace

You can also skip the decorator and use <a href="../tf/py_function.md"><code>tf.py_function</code></a> inplace.
This form can a useful shortcut if you don't control the function's source,
but it is harder to read.

```
>>> # No decorator
>>> def log_huber(x, m):
...   if tf.abs(x) <= m:
...     return x**2
...   else:
...     return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))
>>>
>>> x = tf.constant(1.0)
>>> m = tf.constant(2.0)
>>>
>>> tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32).numpy()
1.0
```

### More info

You can also use <a href="../tf/py_function.md"><code>tf.py_function</code></a> to debug your models at runtime
using Python tools, i.e., you can isolate portions of your code that
you want to debug, wrap them in Python functions and insert `pdb` tracepoints
or print statements as desired, and wrap those functions in
<a href="../tf/py_function.md"><code>tf.py_function</code></a>.

For more information on eager execution, see the
[Eager guide](https://tensorflow.org/guide/eager).

<a href="../tf/py_function.md"><code>tf.py_function</code></a> is similar in spirit to <a href="../tf/numpy_function.md"><code>tf.numpy_function</code></a>, but unlike
the latter, the former lets you use TensorFlow operations in the wrapped
Python function. In particular, while <a href="../tf/compat/v1/py_func.md"><code>tf.compat.v1.py_func</code></a> only runs on CPUs
and wraps functions that take NumPy arrays as inputs and return NumPy arrays
as outputs, <a href="../tf/py_function.md"><code>tf.py_function</code></a> can be placed on GPUs and wraps functions
that take Tensors as inputs, execute TensorFlow operations in their bodies,
and return Tensors as outputs.

Note: We recommend to avoid using <a href="../tf/py_function.md"><code>tf.py_function</code></a> outside of prototyping
and experimentation due to the following known limitations:

* Calling <a href="../tf/py_function.md"><code>tf.py_function</code></a> will acquire the Python Global Interpreter Lock
  (GIL) that allows only one thread to run at any point in time. This will
  preclude efficient parallelization and distribution of the execution of the
  program.

* The body of the function (i.e. `func`) will not be serialized in a
  `GraphDef`. Therefore, you should not use this function if you need to
  serialize your model and restore it in a different environment.

* The operation must run in the same address space as the Python program
  that calls <a href="../tf/py_function.md"><code>tf.py_function()</code></a>. If you are using distributed
  TensorFlow, you must run a <a href="../tf/distribute/Server.md"><code>tf.distribute.Server</code></a> in the same process as the
  program that calls <a href="../tf/py_function.md"><code>tf.py_function()</code></a> and you must pin the created
  operation to a device in that server (e.g. using `with tf.device():`).

* Currently <a href="../tf/py_function.md"><code>tf.py_function</code></a> is not compatible with XLA. Calling
  <a href="../tf/py_function.md"><code>tf.py_function</code></a> inside <a href="../tf/function.md"><code>tf.function(jit_compile=True)</code></a> will raise an
  error.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`func`<a id="func"></a>
</td>
<td>
A Python function that accepts `inp` as arguments, and returns a value
(or list of values) whose type is described by `Tout`. Do not set `func`
when using <a href="../tf/py_function.md"><code>tf.py_function</code></a> as a decorator.
</td>
</tr><tr>
<td>
`inp`<a id="inp"></a>
</td>
<td>
Input arguments for `func`.  A list whose elements are `Tensor`s or
`CompositeTensors` (such as <a href="../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>); or a single `Tensor` or
`CompositeTensor`. Do not set `inp` when using <a href="../tf/py_function.md"><code>tf.py_function</code></a> as a
decorator.
</td>
</tr><tr>
<td>
`Tout`<a id="Tout"></a>
</td>
<td>
The type(s) of the value(s) returned by `func`.  One of the following.
* If `func` returns a `Tensor` (or a value that can be converted to a
  Tensor): the <a href="../tf/dtypes/DType.md"><code>tf.DType</code></a> for that value. 
* If `func` returns a `CompositeTensor`: The <a href="../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> for that value.
* If `func` returns `None`: the empty list (`[]`). 
* If `func` returns a list of `Tensor` and `CompositeTensor` values: a
  corresponding list of <a href="../tf/dtypes/DType.md"><code>tf.DType</code></a>s and <a href="../tf/TypeSpec.md"><code>tf.TypeSpec</code></a>s for each value.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
* If `func` is `None` this returns a decorator that will ensure the
decorated function will always run with eager execution even if called
from a <a href="../tf/function.md"><code>tf.function</code></a>/<a href="../tf/Graph.md"><code>tf.Graph</code></a>.
* If used `func` is not `None` this executes `func` with eager execution
and returns the result: a `Tensor`, `CompositeTensor`, or list of
`Tensor` and `CompositeTensor`; or an empty list if `func` returns `None`.
</td>
</tr>

</table>

