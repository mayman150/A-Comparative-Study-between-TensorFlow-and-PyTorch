description: Returns the current variable scope.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.get_variable_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.get_variable_scope

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/variable_scope.py">View source</a>



Returns the current variable scope.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.get_variable_scope()
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

Although it is a legacy <a href="../../../tf/compat/v1.md"><code>compat.v1</code></a> api,
<a href="../../../tf/compat/v1/get_variable.md"><code>tf.compat.v1.get_variable</code></a> is compatible with eager
execution and <a href="../../../tf/function.md"><code>tf.function</code></a>

However, to maintain variable-scope based variable reuse
you will need to combine it with
`tf.compat.v1.keras.utils.track_tf1_style_variables`. (Though
it will behave as if reuse is always set to <a href="../../../tf/compat/v1.md#AUTO_REUSE"><code>tf.compat.v1.AUTO_REUSE</code></a>.)

See the
[migration guide](https://www.tensorflow.org/guide/migrate/model_mapping)
for more info.

The TF2 equivalent, if you are just trying to track
variable name prefixes and not control `get_variable`-based variable reuse,
would be to use <a href="../../../tf/name_scope.md"><code>tf.name_scope</code></a> and capture the output of opening the
scope (which represents the current name prefix).

For example:
```python
x = tf.name_scope('foo') as current_scope:
  ...
```

 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->
