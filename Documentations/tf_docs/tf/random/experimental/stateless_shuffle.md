description: Randomly and deterministically shuffles a tensor along its first dimension.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.experimental.stateless_shuffle" />
<meta itemprop="path" content="Stable" />
</div>

# tf.random.experimental.stateless_shuffle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/stateless_random_ops.py">View source</a>



Randomly and deterministically shuffles a tensor along its first dimension.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.experimental.stateless_shuffle`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.random.experimental.stateless_shuffle(
    value, seed, alg=&#x27;auto_select&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
to one and only one `output[i]`. For example, a mapping that might occur for a
3x2 tensor is:

```python
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```

```
>>> v = tf.constant([[1, 2], [3, 4], [5, 6]])
>>> shuffled = tf.random.experimental.stateless_shuffle(v, seed=[8, 9])
>>> print(shuffled)
tf.Tensor(
[[5 6]
  [1 2]
  [3 4]], shape=(3, 2), dtype=int32)
```

This is a stateless version of <a href="../../../tf/random/shuffle.md"><code>tf.random.shuffle</code></a>: if run twice with the
same `value` and `seed`, it will produce the same result.  The
output is consistent across multiple runs on the same hardware (and between
CPU and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
hardware.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`<a id="value"></a>
</td>
<td>
A Tensor to be shuffled.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
A shape [2] Tensor. The seed to the random number generator. Must have
dtype `int32` or `int64`.
</td>
</tr><tr>
<td>
`alg`<a id="alg"></a>
</td>
<td>
The RNG algorithm used to generate the random numbers. See
<a href="../../../tf/random/stateless_uniform.md"><code>tf.random.stateless_uniform</code></a> for a detailed explanation.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of same shape and type as `value`, shuffled along its first
dimension.
</td>
</tr>

</table>

