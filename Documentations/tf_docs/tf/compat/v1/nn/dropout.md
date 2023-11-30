description: Computes dropout. (deprecated arguments)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.dropout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.nn.dropout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Computes dropout. (deprecated arguments)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.dropout(
    x, keep_prob=None, noise_shape=None, seed=None, name=None, rate=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: SOME ARGUMENTS ARE DEPRECATED: `(keep_prob)`. They will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.

For each element of `x`, with probability `rate`, outputs `0`, and otherwise
scales up the input by `1 / (1-rate)`. The scaling is such that the expected
sum is unchanged.

By default, each element is kept or dropped independently.  If `noise_shape`
is specified, it must be
[broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
kept independently and each row and column will be kept or not kept together.

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
`keep_prob`<a id="keep_prob"></a>
</td>
<td>
(deprecated) A deprecated alias for `(1-rate)`.
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
`seed`<a id="seed"></a>
</td>
<td>
A Python integer. Used to create random seeds. See
<a href="../../../../tf/random/set_seed.md"><code>tf.random.set_seed</code></a> for behavior.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this operation (optional).
</td>
</tr><tr>
<td>
`rate`<a id="rate"></a>
</td>
<td>
A scalar `Tensor` with the same type as `x`. The probability that each
element of `x` is discarded.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor of the same shape of `x`.
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
If `rate` is not in `[0, 1)` or if `x` is not a floating
point tensor.
</td>
</tr>
</table>

