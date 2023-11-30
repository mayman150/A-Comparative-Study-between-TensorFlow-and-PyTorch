description: Convert a TensorFlow GraphDef to TFLite. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.lite.toco_convert" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.lite.toco_convert

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/convert.py">View source</a>



Convert a TensorFlow GraphDef to TFLite. (deprecated)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.lite.toco_convert(
    input_data, input_tensors, output_tensors, *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `lite.TFLiteConverter` instead.

This function is deprecated. Please use <a href="../../../../tf/lite/TFLiteConverter.md"><code>tf.lite.TFLiteConverter</code></a> API instead.
Conversion can be customized by providing arguments that are forwarded to
`build_model_flags` and `build_conversion_flags` (see documentation for
details).
Args:
  input_data: Input data (i.e. often `sess.graph_def`).
  input_tensors: List of input tensors. Type and shape are computed using
    `foo.shape` and `foo.dtype`.
  output_tensors: List of output tensors (only .name is used from this).
  *args: See `build_model_flags` and `build_conversion_flags`.
  **kwargs: See `build_model_flags` and `build_conversion_flags`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The converted TensorFlow Lite model in a bytes array.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
Defined in `convert`.
</td>
</tr>

</table>

