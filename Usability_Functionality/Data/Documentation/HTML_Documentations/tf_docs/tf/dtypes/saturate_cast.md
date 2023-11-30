description: Performs a safe saturating cast of value to dtype.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.dtypes.saturate_cast" />
<meta itemprop="path" content="Stable" />
</div>

# tf.dtypes.saturate_cast

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Performs a safe saturating cast of `value` to `dtype`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.saturate_cast`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.dtypes.saturate_cast`, `tf.compat.v1.saturate_cast`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.dtypes.saturate_cast(
    value, dtype, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function casts the input to `dtype` without overflow.  If
there is a danger that values would over or underflow in the cast, this op
applies the appropriate clamping before the cast.  See <a href="../../tf/cast.md"><code>tf.cast</code></a> for more
details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`<a id="value"></a>
</td>
<td>
A `Tensor`.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The desired output `DType`.
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
`value` safely cast to `dtype`.
</td>
</tr>

</table>

