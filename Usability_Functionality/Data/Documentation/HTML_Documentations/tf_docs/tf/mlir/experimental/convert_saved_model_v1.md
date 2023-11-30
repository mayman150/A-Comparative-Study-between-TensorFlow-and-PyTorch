description: Converts a v1 SavedModel to MLIR module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.mlir.experimental.convert_saved_model_v1" />
<meta itemprop="path" content="Stable" />
</div>

# tf.mlir.experimental.convert_saved_model_v1

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/mlir/mlir.py">View source</a>



Converts a v1 SavedModel to MLIR module.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.mlir.experimental.convert_saved_model_v1`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.mlir.experimental.convert_saved_model_v1(
    saved_model_path,
    exported_names,
    tags,
    lift_variables,
    include_variables_in_initializers,
    upgrade_legacy=True,
    show_debug_info=False
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
`tags`<a id="tags"></a>
</td>
<td>
MetaGraphDef to be loaded is identified by the supplied tags.
</td>
</tr><tr>
<td>
`lift_variables`<a id="lift_variables"></a>
</td>
<td>
Whether to promote tf.VarHandleOp to resource arguments.
</td>
</tr><tr>
<td>
`include_variables_in_initializers`<a id="include_variables_in_initializers"></a>
</td>
<td>
Keeps the variables in initializers
before lifting variables.
</td>
</tr><tr>
<td>
`upgrade_legacy`<a id="upgrade_legacy"></a>
</td>
<td>
Functionalize the input graph before importing.
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
SavedModule.
</td>
</tr>

</table>

