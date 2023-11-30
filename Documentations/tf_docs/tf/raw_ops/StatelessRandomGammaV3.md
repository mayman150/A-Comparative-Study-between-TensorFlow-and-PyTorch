description: Outputs deterministic pseudorandom random numbers from a gamma distribution.
robots: noindex

# tf.raw_ops.StatelessRandomGammaV3

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Outputs deterministic pseudorandom random numbers from a gamma distribution.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.StatelessRandomGammaV3`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.StatelessRandomGammaV3(
    shape, key, counter, alg, alpha, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Outputs random values from a gamma distribution.

The outputs are a deterministic function of the inputs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
The shape of the output tensor.
</td>
</tr><tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A `Tensor` of type `uint64`.
Key for the counter-based RNG algorithm (shape uint64[1]).
</td>
</tr><tr>
<td>
`counter`<a id="counter"></a>
</td>
<td>
A `Tensor` of type `uint64`.
Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
</td>
</tr><tr>
<td>
`alg`<a id="alg"></a>
</td>
<td>
A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
</td>
</tr><tr>
<td>
`alpha`<a id="alpha"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
The concentration of the gamma distribution. Shape must match the rightmost
dimensions of `shape`.
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
A `Tensor`. Has the same type as `alpha`.
</td>
</tr>

</table>

