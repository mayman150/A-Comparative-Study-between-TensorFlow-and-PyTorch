description: Runs a pipeline over input module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.mlir.experimental.run_pass_pipeline" />
<meta itemprop="path" content="Stable" />
</div>

# tf.mlir.experimental.run_pass_pipeline

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/mlir/mlir.py">View source</a>



Runs a pipeline over input module.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.mlir.experimental.run_pass_pipeline`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.mlir.experimental.run_pass_pipeline(
    mlir_txt, pass_pipeline, show_debug_info=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mlir_txt`<a id="mlir_txt"></a>
</td>
<td>
Textual representation of the MLIR module.
</td>
</tr><tr>
<td>
`pass_pipeline`<a id="pass_pipeline"></a>
</td>
<td>
Pass pipeline to run on module.
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
transformed module.
</td>
</tr>

</table>

