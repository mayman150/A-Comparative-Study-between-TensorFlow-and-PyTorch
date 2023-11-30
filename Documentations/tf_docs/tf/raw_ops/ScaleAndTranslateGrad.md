robots: noindex

# tf.raw_ops.ScaleAndTranslateGrad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>






<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ScaleAndTranslateGrad`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ScaleAndTranslateGrad(
    grads,
    original_image,
    scale,
    translation,
    kernel_type=&#x27;lanczos3&#x27;,
    antialias=True,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`grads`<a id="grads"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`.
</td>
</tr><tr>
<td>
`original_image`<a id="original_image"></a>
</td>
<td>
A `Tensor`. Must have the same type as `grads`.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`translation`<a id="translation"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`kernel_type`<a id="kernel_type"></a>
</td>
<td>
An optional `string`. Defaults to `"lanczos3"`.
</td>
</tr><tr>
<td>
`antialias`<a id="antialias"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
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
A `Tensor`. Has the same type as `grads`.
</td>
</tr>

</table>

