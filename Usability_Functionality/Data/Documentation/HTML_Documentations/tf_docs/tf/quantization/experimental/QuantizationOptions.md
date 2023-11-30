description: A ProtocolMessage

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.quantization.experimental.QuantizationOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="RepresentativeDatasetsEntry"/>
</div>

# tf.quantization.experimental.QuantizationOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.proto">View source</a>



A ProtocolMessage

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.quantization.experimental.QuantizationOptions`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`calibration_options`<a id="calibration_options"></a>
</td>
<td>
`CalibrationOptions calibration_options`
</td>
</tr><tr>
<td>
`debugger_options`<a id="debugger_options"></a>
</td>
<td>
`DebuggerOptions debugger_options`
</td>
</tr><tr>
<td>
`enable_legacy_weight_only`<a id="enable_legacy_weight_only"></a>
</td>
<td>
`bool enable_legacy_weight_only`
</td>
</tr><tr>
<td>
`enable_per_channel_quantization`<a id="enable_per_channel_quantization"></a>
</td>
<td>
`bool enable_per_channel_quantization`
</td>
</tr><tr>
<td>
`enable_two_input_tensors`<a id="enable_two_input_tensors"></a>
</td>
<td>
`bool enable_two_input_tensors`
</td>
</tr><tr>
<td>
`experimental_enable_tpu_model_support`<a id="experimental_enable_tpu_model_support"></a>
</td>
<td>
`bool experimental_enable_tpu_model_support`
</td>
</tr><tr>
<td>
`force_graph_mode_calibration`<a id="force_graph_mode_calibration"></a>
</td>
<td>
`bool force_graph_mode_calibration`
</td>
</tr><tr>
<td>
`freeze_all_variables`<a id="freeze_all_variables"></a>
</td>
<td>
`bool freeze_all_variables`
</td>
</tr><tr>
<td>
`min_num_elements_for_weights`<a id="min_num_elements_for_weights"></a>
</td>
<td>
`int64 min_num_elements_for_weights`
</td>
</tr><tr>
<td>
`op_set`<a id="op_set"></a>
</td>
<td>
`OpSet op_set`
</td>
</tr><tr>
<td>
`quantization_method`<a id="quantization_method"></a>
</td>
<td>
`QuantizationMethod quantization_method`
</td>
</tr><tr>
<td>
`representative_datasets`<a id="representative_datasets"></a>
</td>
<td>
`repeated RepresentativeDatasetsEntry representative_datasets`
</td>
</tr><tr>
<td>
`signature_keys`<a id="signature_keys"></a>
</td>
<td>
`repeated string signature_keys`
</td>
</tr><tr>
<td>
`tags`<a id="tags"></a>
</td>
<td>
`repeated string tags`
</td>
</tr><tr>
<td>
`unit_wise_quantization_specs`<a id="unit_wise_quantization_specs"></a>
</td>
<td>
`repeated UnitWiseQuantizationSpec unit_wise_quantization_specs`
</td>
</tr>
</table>



## Child Classes
[`class RepresentativeDatasetsEntry`](../../../tf/quantization/experimental/QuantizationOptions/RepresentativeDatasetsEntry.md)

