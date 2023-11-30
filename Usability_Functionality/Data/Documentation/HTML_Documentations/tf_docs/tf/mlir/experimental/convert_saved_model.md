description: Converts a SavedModel to MLIR module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.mlir.experimental.convert_saved_model" />
<meta itemprop="path" content="Stable" />
</div>

# tf.mlir.experimental.convert_saved_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/mlir/mlir.py">View source</a>



Converts a SavedModel to MLIR module.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.mlir.experimental.convert_saved_model`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.mlir.experimental.convert_saved_model(
    saved_model_path, exported_names, show_debug_info=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`saved_model_path`<a id="saved_model_path"></a>
</td>
<td>
Path to SavedModel.
</td>
</tr><tr>
<td>
`exported_names`<a id="exported_names"></a>
</td>
<td>
Names to export.
</td>
</tr><tr>
<td>
`show_debug_info`<a id="show_debug_info"></a>
</td>
<td>
Whether to include locations in the emitted textual form.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A textual representation of the MLIR module corresponding to the
SavedModel.
</td>
</tr>

</table>

