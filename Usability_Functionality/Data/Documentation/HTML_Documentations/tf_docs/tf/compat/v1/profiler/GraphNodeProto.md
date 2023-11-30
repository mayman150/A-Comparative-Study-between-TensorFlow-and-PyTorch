description: A ProtocolMessage

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.profiler.GraphNodeProto" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="InputShapesEntry"/>
</div>

# tf.compat.v1.profiler.GraphNodeProto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/profiler/tfprof_output.proto">View source</a>



A ProtocolMessage

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`accelerator_exec_micros`<a id="accelerator_exec_micros"></a>
</td>
<td>
`int64 accelerator_exec_micros`
</td>
</tr><tr>
<td>
`children`<a id="children"></a>
</td>
<td>
`repeated GraphNodeProto children`
</td>
</tr><tr>
<td>
`cpu_exec_micros`<a id="cpu_exec_micros"></a>
</td>
<td>
`int64 cpu_exec_micros`
</td>
</tr><tr>
<td>
`devices`<a id="devices"></a>
</td>
<td>
`repeated string devices`
</td>
</tr><tr>
<td>
`exec_micros`<a id="exec_micros"></a>
</td>
<td>
`int64 exec_micros`
</td>
</tr><tr>
<td>
`float_ops`<a id="float_ops"></a>
</td>
<td>
`int64 float_ops`
</td>
</tr><tr>
<td>
`input_shapes`<a id="input_shapes"></a>
</td>
<td>
`repeated InputShapesEntry input_shapes`
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
`string name`
</td>
</tr><tr>
<td>
`output_bytes`<a id="output_bytes"></a>
</td>
<td>
`int64 output_bytes`
</td>
</tr><tr>
<td>
`parameters`<a id="parameters"></a>
</td>
<td>
`int64 parameters`
</td>
</tr><tr>
<td>
`peak_bytes`<a id="peak_bytes"></a>
</td>
<td>
`int64 peak_bytes`
</td>
</tr><tr>
<td>
`requested_bytes`<a id="requested_bytes"></a>
</td>
<td>
`int64 requested_bytes`
</td>
</tr><tr>
<td>
`residual_bytes`<a id="residual_bytes"></a>
</td>
<td>
`int64 residual_bytes`
</td>
</tr><tr>
<td>
`run_count`<a id="run_count"></a>
</td>
<td>
`int64 run_count`
</td>
</tr><tr>
<td>
`shapes`<a id="shapes"></a>
</td>
<td>
`repeated TensorShapeProto shapes`
</td>
</tr><tr>
<td>
`tensor_value`<a id="tensor_value"></a>
</td>
<td>
`TFProfTensorProto tensor_value`
</td>
</tr><tr>
<td>
`total_accelerator_exec_micros`<a id="total_accelerator_exec_micros"></a>
</td>
<td>
`int64 total_accelerator_exec_micros`
</td>
</tr><tr>
<td>
`total_cpu_exec_micros`<a id="total_cpu_exec_micros"></a>
</td>
<td>
`int64 total_cpu_exec_micros`
</td>
</tr><tr>
<td>
`total_definition_count`<a id="total_definition_count"></a>
</td>
<td>
`int64 total_definition_count`
</td>
</tr><tr>
<td>
`total_exec_micros`<a id="total_exec_micros"></a>
</td>
<td>
`int64 total_exec_micros`
</td>
</tr><tr>
<td>
`total_float_ops`<a id="total_float_ops"></a>
</td>
<td>
`int64 total_float_ops`
</td>
</tr><tr>
<td>
`total_output_bytes`<a id="total_output_bytes"></a>
</td>
<td>
`int64 total_output_bytes`
</td>
</tr><tr>
<td>
`total_parameters`<a id="total_parameters"></a>
</td>
<td>
`int64 total_parameters`
</td>
</tr><tr>
<td>
`total_peak_bytes`<a id="total_peak_bytes"></a>
</td>
<td>
`int64 total_peak_bytes`
</td>
</tr><tr>
<td>
`total_requested_bytes`<a id="total_requested_bytes"></a>
</td>
<td>
`int64 total_requested_bytes`
</td>
</tr><tr>
<td>
`total_residual_bytes`<a id="total_residual_bytes"></a>
</td>
<td>
`int64 total_residual_bytes`
</td>
</tr><tr>
<td>
`total_run_count`<a id="total_run_count"></a>
</td>
<td>
`int64 total_run_count`
</td>
</tr>
</table>



## Child Classes
[`class InputShapesEntry`](../../../../tf/compat/v1/profiler/GraphNodeProto/InputShapesEntry.md)

