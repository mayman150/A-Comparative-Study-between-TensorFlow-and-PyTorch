description: Defines a function as a recompute-checkpoint for the tape auto-diff.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.recompute_grad" />
<meta itemprop="path" content="Stable" />
</div>

# tf.recompute_grad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/custom_gradient.py">View source</a>



Defines a function as a recompute-checkpoint for the tape auto-diff.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.recompute_grad`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.recompute_grad(
    f
)
</code></pre>



<!-- Placeholder for "Used in" -->

Tape checkpointing is a technique to reduce the memory consumption of the
auto-diff tape:

- Without tape checkpointing operations and intermediate values are
recorded to the tape for use in the backward pass.

- With tape checkpointing, only the function call and its inputs are
recorded. During back-propagation the `recompute_grad` custom gradient
(<a href="../tf/custom_gradient.md"><code>tf.custom_gradient</code></a>) recomputes the function under a localized Tape object.
This recomputation of the function during backpropagation performs redundant
calculation, but reduces the overall memory usage of the Tape.

```
>>> y = tf.Variable(1.0)
```

```
>>> def my_function(x):
...   tf.print('running')
...   z = x*y
...   return z
```

```
>>> my_function_recompute = tf.recompute_grad(my_function)
```

```
>>> with tf.GradientTape() as tape:
...   r = tf.constant(1.0)
...   for i in range(4):
...     r = my_function_recompute(r)
running
running
running
running
```

```
>>> grad = tape.gradient(r, [y])
running
running
running
running
```

Without `recompute_grad`, the tape contains all intermitate steps, and no
recomputation is performed.

```
>>> with tf.GradientTape() as tape:
...   r = tf.constant(1.0)
...   for i in range(4):
...     r = my_function(r)
running
running
running
running
```

```
>>> grad = tape.gradient(r, [y])
```


If `f` was a <a href="../tf/keras.md"><code>tf.keras</code></a> `Model` or `Layer` object, methods and attributes
such as `f.variables` are not available on the returned function `g`.
Either keep a reference of `f` , or use `g.__wrapped__` for accessing
these variables and methods.


```
>>> def print_running_and_return(x):
...   tf.print("running")
...   return x
```

```
>>> model = tf.keras.Sequential([
...   tf.keras.layers.Lambda(print_running_and_return),
...   tf.keras.layers.Dense(2)
... ])
```

```
>>> model_recompute = tf.recompute_grad(model)
```

```
>>> with tf.GradientTape(persistent=True) as tape:
...   r = tf.constant([[1,2]])
...   for i in range(4):
...     r = model_recompute(r)
running
running
running
running
```

```
>>> grad = tape.gradient(r, model.variables)
running
running
running
running
```

Alternatively, use the `__wrapped__` attribute to access the original
model object.

```
>>> grad = tape.gradient(r, model_recompute.__wrapped__.variables)
running
running
running
running
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`f`<a id="f"></a>
</td>
<td>
function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A function `g` wrapping `f` that defines a custom gradient, which recomputes
`f` on the backwards pass of a gradient call.
</td>
</tr>

</table>

