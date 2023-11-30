description: Computes Fresnel's sine integral of x element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.math.special.fresnel_sin" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.math.special.fresnel_sin

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/special_math_ops.py">View source</a>



Computes Fresnel's sine integral of `x` element-wise.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.math.special.fresnel_sin(
    x, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The Fresnel sine integral is defined as the integral of `sin(t^2)` from
`0` to `x`, with the domain of definition all real numbers.

```
>>> tf.math.special.fresnel_sin([-1., -0.1, 0.1, 1.]).numpy()
array([-0.43825912, -0.00052359,  0.00052359,  0.43825912], dtype=float32)
```

This implementation is based off of the Cephes math library.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
A `Tensor` or `SparseTensor`. Must be one of the following types:
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
A `Tensor` or `SparseTensor`, respectively. Has the same type as `x`.
</td>
</tr>

</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">scipy compatibility</h2>

Equivalent to scipy.special.fresnel first output.

 </devsite-expandable></section>

