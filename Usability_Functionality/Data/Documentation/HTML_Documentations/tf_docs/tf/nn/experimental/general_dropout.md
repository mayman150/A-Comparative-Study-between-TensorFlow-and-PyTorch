description: Computes dropout: randomly sets elements to zero to prevent overfitting.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.experimental.general_dropout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.experimental.general_dropout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Computes dropout: randomly sets elements to zero to prevent overfitting.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.nn.experimental.general_dropout`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.experimental.general_dropout(
    x, rate, uniform_sampler, noise_shape=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Please see <a href="../../../tf/nn/experimental/stateless_dropout.md"><code>tf.nn.experimental.stateless_dropout</code></a> for an overview
of dropout.

Unlike <a href="../../../tf/nn/experimental/stateless_dropout.md"><code>tf.nn.experimental.stateless_dropout</code></a>, here you can supply a
custom sampler function `uniform_sampler` that (given a shape and a
dtype) generates a random, `Uniform[0, 1)`-distributed tensor (of
that shape and dtype).  `uniform_sampler` can be
e.g. `tf.random.stateless_random_uniform` or
<a href="../../../tf/random/Generator.md#uniform"><code>tf.random.Generator.uniform</code></a>.

For example, if you are using <a href="../../../tf/random/Generator.md"><code>tf.random.Generator</code></a> to generate
random numbers, you can use this code to do dropouts:

```
>>> g = tf.random.Generator.from_seed(7)
>>> sampler = g.uniform
>>> x = tf.constant([1.1, 2.2, 3.3, 4.4, 5.5])
>>> rate = 0.5
>>> tf.nn.experimental.general_dropout(x, rate, sampler)
<tf.Tensor: shape=(5,), ..., numpy=array([ 0. ,  4.4,  6.6,  8.8, 11. ], ...)>
>>> tf.nn.experimental.general_dropout(x, rate, sampler)
<tf.Tensor: shape=(5,), ..., numpy=array([2.2, 0. , 0. , 8.8, 0. ], ...)>
```

It has better performance than using
<a href="../../../tf/nn/experimental/stateless_dropout.md"><code>tf.nn.experimental.stateless_dropout</code></a> and
<a href="../../../tf/random/Generator.md#make_seeds"><code>tf.random.Generator.make_seeds</code></a>:

```
>>> g = tf.random.Generator.from_seed(7)
>>> x = tf.constant([1.1, 2.2, 3.3, 4.4, 5.5])
>>> rate = 0.5
>>> tf.nn.experimental.stateless_dropout(x, rate, g.make_seeds(1)[:, 0])
<tf.Tensor: shape=(5,), ..., numpy=array([ 2.2,  4.4,  6.6,  0. , 11. ], ...)>
>>> tf.nn.experimental.stateless_dropout(x, rate, g.make_seeds(1)[:, 0])
<tf.Tensor: shape=(5,), ..., numpy=array([2.2, 0. , 6.6, 8.8, 0. ], ...>
```

because generating and consuming seeds cost extra
computation. <a href="../../../tf/nn/experimental/general_dropout.md"><code>tf.nn.experimental.general_dropout</code></a> can let you avoid
them.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
A floating point tensor.
</td>
</tr><tr>
<td>
`rate`<a id="rate"></a>
</td>
<td>
A scalar `Tensor` with the same type as x. The probability
that each element is dropped. For example, setting rate=0.1 would drop
10% of input elements.
</td>
</tr><tr>
<td>
`uniform_sampler`<a id="uniform_sampler"></a>
</td>
<td>
a callable of signature `(shape, dtype) ->
Tensor[shape, dtype]`, used to generate a tensor of uniformly-distributed
random numbers in the range `[0, 1)`, of the given shape and dtype.
</td>
</tr><tr>
<td>
`noise_shape`<a id="noise_shape"></a>
</td>
<td>
A 1-D integer `Tensor`, representing the
shape for randomly generated keep/drop flags.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor of the same shape and dtype of `x`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If `rate` is not in `[0, 1)` or if `x` is not a floating point
tensor. `rate=1` is disallowed, because the output would be all zeros,
which is likely not what was intended.
</td>
</tr>
</table>

