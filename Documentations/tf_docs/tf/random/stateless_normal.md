description: Outputs deterministic pseudorandom values from a normal distribution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.stateless_normal" />
<meta itemprop="path" content="Stable" />
</div>

# tf.random.stateless_normal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/stateless_random_ops.py">View source</a>



Outputs deterministic pseudorandom values from a normal distribution.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.stateless_normal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.random.stateless_normal(
    shape,
    seed,
    mean=0.0,
    stddev=1.0,
    dtype=<a href="../../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    name=None,
    alg=&#x27;auto_select&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is a stateless version of <a href="../../tf/random/normal.md"><code>tf.random.normal</code></a>: if run twice with the
same seeds and shapes, it will produce the same pseudorandom numbers.  The
output is consistent across multiple runs on the same hardware (and between
CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
hardware.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A 1-D integer Tensor or Python array. The shape of the output tensor.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A shape [2] Tensor, the seed to the random number generator. Must have
dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
</td>
</tr><tr>
<td>
`mean`<a id="mean"></a>
</td>
<td>
A 0-D Tensor or Python value of type `dtype`. The mean of the normal
distribution.
</td>
</tr><tr>
<td>
`stddev`<a id="stddev"></a>
</td>
<td>
A 0-D Tensor or Python value of type `dtype`. The standard deviation
of the normal distribution.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The float type of the output: `float16`, `bfloat16`, `float32`,
`float64`. Defaults to `float32`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`alg`<a id="alg"></a>
</td>
<td>
The RNG algorithm used to generate the random numbers. See
<a href="../../tf/random/stateless_uniform.md"><code>tf.random.stateless_uniform</code></a> for a detailed explanation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of the specified shape filled with random normal values.
</td>
</tr>

</table>

