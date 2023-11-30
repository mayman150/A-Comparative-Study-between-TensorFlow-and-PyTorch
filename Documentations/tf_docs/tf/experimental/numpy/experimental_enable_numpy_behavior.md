description: Enable NumPy behavior on Tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.experimental_enable_numpy_behavior" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.numpy.experimental_enable_numpy_behavior

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/numpy_ops/np_config.py">View source</a>



Enable NumPy behavior on Tensors.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.experimental_enable_numpy_behavior(
    prefer_float32=False, dtype_conversion_mode=&#x27;legacy&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Enabling NumPy behavior has three effects:
* It adds to <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> some common NumPy methods such as `T`,
  `reshape` and `ravel`.
* It changes dtype promotion in <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a> operators to be
  compatible with NumPy. For example,
  `tf.ones([], tf.int32) + tf.ones([], tf.float32)` used to throw a
  "dtype incompatible" error, but after this it will return a
  float64 tensor (obeying NumPy's promotion rules).
* It enhances <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>'s indexing capability to be on par with
  [NumPy's](https://numpy.org/doc/stable/reference/arrays.indexing.html).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`prefer_float32`<a id="prefer_float32"></a>
</td>
<td>
Controls whether dtype inference will use float32 for Python
floats, or float64 (the default and the NumPy-compatible behavior).
</td>
</tr><tr>
<td>
`dtype_conversion_mode`<a id="dtype_conversion_mode"></a>
</td>
<td>
a string that specifies promotion mode. This string
corresponds to a PromoMode Enum and can be 'off', 'legacy', 'safe', or
'all'. 'safe' or 'all' mode enables the auto dtype conversion semantics.
</td>
</tr>
</table>

