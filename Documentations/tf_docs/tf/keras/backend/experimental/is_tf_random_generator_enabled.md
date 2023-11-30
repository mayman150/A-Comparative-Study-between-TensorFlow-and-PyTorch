description: Check whether <a href="../../../../tf/random/Generator.md"><code>tf.random.Generator</code></a> is used for RNG in Keras.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.backend.experimental.is_tf_random_generator_enabled" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.backend.experimental.is_tf_random_generator_enabled

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/backend.py#L1831-L1860">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check whether <a href="../../../../tf/random/Generator.md"><code>tf.random.Generator</code></a> is used for RNG in Keras.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.backend.experimental.is_tf_random_generator_enabled()
</code></pre>



<!-- Placeholder for "Used in" -->

Compared to existing TF stateful random ops, <a href="../../../../tf/random/Generator.md"><code>tf.random.Generator</code></a> uses
<a href="../../../../tf/Variable.md"><code>tf.Variable</code></a> and stateless random ops to generate random numbers,
which leads to better reproducibility in distributed training.
Note enabling it might introduce some breakage to existing code,
by producing differently-seeded random number sequences
and breaking tests that rely on specific random numbers being generated.
To disable the
usage of <a href="../../../../tf/random/Generator.md"><code>tf.random.Generator</code></a>, please use
`tf.keras.backend.experimental.disable_random_generator`.

We expect the <a href="../../../../tf/random/Generator.md"><code>tf.random.Generator</code></a> code path to become the default, and
will remove the legacy stateful random ops such as <a href="../../../../tf/random/uniform.md"><code>tf.random.uniform</code></a> in
the future (see the [TF RNG guide](
https://www.tensorflow.org/guide/random_numbers)).

This API will also be removed in a future release as well, together with
<a href="../../../../tf/keras/backend/experimental/enable_tf_random_generator.md"><code>tf.keras.backend.experimental.enable_tf_random_generator()</code></a> and
<a href="../../../../tf/keras/backend/experimental/disable_tf_random_generator.md"><code>tf.keras.backend.experimental.disable_tf_random_generator()</code></a>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`boolean`<a id="boolean"></a>
</td>
<td>
whether <a href="../../../../tf/random/Generator.md"><code>tf.random.Generator</code></a> is used for random number
generation in Keras.
</td>
</tr>
</table>

