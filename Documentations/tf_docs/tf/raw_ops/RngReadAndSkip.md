description: Advance the counter of a counter-based RNG.
robots: noindex

# tf.raw_ops.RngReadAndSkip

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Advance the counter of a counter-based RNG.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.RngReadAndSkip`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.RngReadAndSkip(
    resource, alg, delta, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The state of the RNG after
`rng_read_and_skip(n)` will be the same as that after `uniform([n])`
(or any other distribution). The actual increment added to the
counter is an unspecified implementation choice.

In the case that the input algorithm is RNG_ALG_AUTO_SELECT, the counter in the state needs to be of size int64[2], the current maximal counter size among algorithms. In this case, this op will manage the counter as if it is an 128-bit integer with layout [lower_64bits, higher_64bits]. If an algorithm needs less than 128 bits for the counter, it should use the left portion of the int64[2]. In this way, the int64[2] is compatible with all current RNG algorithms (Philox, ThreeFry and xla::RandomAlgorithm::RNG_DEFAULT). Downstream RNG ops can thus use this counter with any RNG algorithm.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`resource`<a id="resource"></a>
</td>
<td>
A `Tensor` of type `resource`.
The handle of the resource variable that stores the state of the RNG. The state consists of the counter followed by the key.
</td>
</tr><tr>
<td>
`alg`<a id="alg"></a>
</td>
<td>
A `Tensor` of type `int32`. The RNG algorithm.
</td>
</tr><tr>
<td>
`delta`<a id="delta"></a>
</td>
<td>
A `Tensor` of type `uint64`. The amount of advancement.
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
A `Tensor` of type `int64`.
</td>
</tr>

</table>

