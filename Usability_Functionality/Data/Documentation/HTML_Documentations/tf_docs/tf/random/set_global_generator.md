description: Replaces the global generator with another Generator object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.set_global_generator" />
<meta itemprop="path" content="Stable" />
</div>

# tf.random.set_global_generator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/stateful_random_ops.py">View source</a>



Replaces the global generator with another `Generator` object.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.random.experimental.set_global_generator`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.experimental.set_global_generator`, `tf.compat.v1.random.set_global_generator`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.random.set_global_generator(
    generator
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function replaces the global generator with the provided `generator`
object.
A random number generator utilizes a <a href="../../tf/Variable.md"><code>tf.Variable</code></a> object to store its state.
The user shall be aware of caveats how `set_global_generator` interacts with
<a href="../../tf/function.md"><code>tf.function</code></a>:

- tf.function puts restrictions on Variable creation thus one cannot freely
  create a new random generator instance inside <a href="../../tf/function.md"><code>tf.function</code></a>.
  To call `set_global_generator` inside <a href="../../tf/function.md"><code>tf.function</code></a>, the generator instance
  must have already been created eagerly.
- tf.function captures the Variable during trace-compilation, thus a compiled
  f.function will not be affected `set_global_generator` as demonstrated by
  random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .

For most use cases, avoid calling `set_global_generator` after program
initialization, and prefer to reset the state of the existing global generator
instead, such as,

```
>>> rng = tf.random.get_global_generator()
>>> rng.reset_from_seed(30)
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`generator`<a id="generator"></a>
</td>
<td>
the new `Generator` object.
</td>
</tr>
</table>

