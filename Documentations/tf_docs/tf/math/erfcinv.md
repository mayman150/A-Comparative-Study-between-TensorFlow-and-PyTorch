description: Computes the inverse of complementary error function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.erfcinv" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.erfcinv

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes the inverse of complementary error function.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.erfcinv(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given `x`, compute the inverse complementary error function of `x`.
This function is the inverse of <a href="../../tf/math/erfc.md"><code>tf.math.erfc</code></a>, and is defined on
`[0, 2]`.

```
>>> tf.math.erfcinv([0., 0.2, 1., 1.5, 2.])
<tf.Tensor: shape=(5,), dtype=float32, numpy=
array([       inf,  0.9061935, -0.       , -0.4769363,       -inf],
      dtype=float32)>
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
`Tensor` with type `float` or `double`.
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
Inverse complementary error function of `x`.
</td>
</tr>

</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">numpy compatibility</h2>

Equivalent to scipy.special.erfcinv

 </devsite-expandable></section>

