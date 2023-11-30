description: Computes x / y.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.xdivy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.xdivy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes `x / y`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.math.xdivy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.xdivy(
    x, y, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given `x` and `y`, computes `x / y`. This function safely returns
zero when `x = 0`, no matter what the value of `y` is.

#### Example:



```
>>> tf.math.xdivy(1., 2.)
<tf.Tensor: shape=(), dtype=float32, numpy=0.5>
>>> tf.math.xdivy(0., 1.)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
>>> tf.math.xdivy(0., 0.)
<tf.Tensor: shape=(), dtype=float32, numpy=0.0>
>>> tf.math.xdivy(1., 0.)
<tf.Tensor: shape=(), dtype=float32, numpy=inf>
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
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> of type `half`, `float32`, `float64`, `complex64`,
`complex128`
</td>
</tr><tr>
<td>
`y`<a id="y"></a>
</td>
<td>
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> of type `half`, `float32`, `float64`, `complex64`,
`complex128`
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
`x / y`.
</td>
</tr>

</table>

