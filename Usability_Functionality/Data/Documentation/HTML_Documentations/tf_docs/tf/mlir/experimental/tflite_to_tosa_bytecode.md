description: Converts TFLite flatbuffer to TOSA dialect in MLIR bytecode.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.mlir.experimental.tflite_to_tosa_bytecode" />
<meta itemprop="path" content="Stable" />
</div>

# tf.mlir.experimental.tflite_to_tosa_bytecode

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/mlir/mlir.py">View source</a>



Converts TFLite flatbuffer to TOSA dialect in MLIR bytecode.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.mlir.experimental.tflite_to_tosa_bytecode`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.mlir.experimental.tflite_to_tosa_bytecode(
    flatbuffer,
    bytecode,
    use_external_constant=False,
    ordered_input_arrays=None,
    ordered_output_arrays=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flatbuffer`<a id="flatbuffer"></a>
</td>
<td>
Path to flatbuffer.
</td>
</tr><tr>
<td>
`bytecode`<a id="bytecode"></a>
</td>
<td>
Path to output bytecode.
</td>
</tr><tr>
<td>
`use_external_constant`<a id="use_external_constant"></a>
</td>
<td>
Whether to create `tfl.external_const` instead of
`tfl.const`.
</td>
</tr><tr>
<td>
`ordered_input_arrays`<a id="ordered_input_arrays"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`ordered_output_arrays`<a id="ordered_output_arrays"></a>
</td>
<td>
If ordered_output_arrays is not empty, then the
function will only return nodes in ordered_output_arrays in the same order
</td>
</tr>
</table>

