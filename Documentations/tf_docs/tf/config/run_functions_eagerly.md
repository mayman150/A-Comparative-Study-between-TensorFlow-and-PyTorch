description: Enables / disables eager execution of <a href="../../tf/function.md"><code>tf.function</code></a>s.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.run_functions_eagerly" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.run_functions_eagerly

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/eager/polymorphic_function/eager_function_run.py">View source</a>



Enables / disables eager execution of <a href="../../tf/function.md"><code>tf.function</code></a>s.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.run_functions_eagerly`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.run_functions_eagerly(
    run_eagerly
)
</code></pre>



<!-- Placeholder for "Used in" -->

Calling <a href="../../tf/config/run_functions_eagerly.md"><code>tf.config.run_functions_eagerly(True)</code></a> will make all
invocations of <a href="../../tf/function.md"><code>tf.function</code></a> run eagerly instead of running as a traced graph
function. This can be useful for debugging. As the code now runs line-by-line,
you can add arbitrary `print` messages or pdb breakpoints to monitor the
inputs/outputs of each Tensorflow operation. However, you should avoid using
this for actual production because it significantly slows down execution.

```
>>> def my_func(a):
...  print(f'a: {a}')
...  return a + a
>>> a_fn = tf.function(my_func)
```

```
>>> # A side effect the first time the function is traced
>>> # In tracing time, `a` is printed with shape and dtype only
>>> a_fn(tf.constant(1))
a: Tensor("a:0", shape=(), dtype=int32)
<tf.Tensor: shape=(), dtype=int32, numpy=2>
```

```
>>> # `print` is a python side effect, it won't execute as the traced function
>>> # is called
>>> a_fn(tf.constant(2))
<tf.Tensor: shape=(), dtype=int32, numpy=4>
```

```
>>> # Now, switch to eager running
>>> tf.config.run_functions_eagerly(True)
>>> # The code now runs eagerly and the actual value of `a` is printed
>>> a_fn(tf.constant(2))
a: 2
<tf.Tensor: shape=(), dtype=int32, numpy=4>
```

```
>>> # Turn this back off
>>> tf.config.run_functions_eagerly(False)
```

Note: This flag has no effect on functions passed into tf.data transformations
as arguments. tf.data functions are never executed eagerly and are always
executed as a compiled Tensorflow Graph.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`run_eagerly`<a id="run_eagerly"></a>
</td>
<td>
Boolean. Whether to run functions eagerly.
</td>
</tr>
</table>

