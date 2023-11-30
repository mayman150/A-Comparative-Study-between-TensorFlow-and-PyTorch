description: Returns element-wise largest integer not greater than x.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.math.floor" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.math.floor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Returns element-wise largest integer not greater than x.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.math.floor(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Both input range is `(-inf, inf)` and the
output range consists of all integer values.

#### For example:



```
>>> x = tf.constant([1.3324, -1.5, 5.555, -2.532, 0.99, float("inf")])
>>> tf.floor(x).numpy()
array([ 1., -2.,  5., -3.,  0., inf], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
 A `Tensor`. Must be one of the following types: `bfloat16`, `half`,
`float32`, `float64`.
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
A `Tensor`. Has the same type as x.
</td>
</tr>

</table>

