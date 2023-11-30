description: A ProtocolMessage

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.ConfigProto" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DeviceCountEntry"/>
<meta itemprop="property" content="Experimental"/>
</div>

# tf.compat.v1.ConfigProto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/protobuf/config.proto">View source</a>



A ProtocolMessage

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`allow_soft_placement`<a id="allow_soft_placement"></a>
</td>
<td>
`bool allow_soft_placement`
</td>
</tr><tr>
<td>
`cluster_def`<a id="cluster_def"></a>
</td>
<td>
`ClusterDef cluster_def`
</td>
</tr><tr>
<td>
`device_count`<a id="device_count"></a>
</td>
<td>
`repeated DeviceCountEntry device_count`
</td>
</tr><tr>
<td>
`device_filters`<a id="device_filters"></a>
</td>
<td>
`repeated string device_filters`
</td>
</tr><tr>
<td>
`experimental`<a id="experimental"></a>
</td>
<td>
`Experimental experimental`
</td>
</tr><tr>
<td>
`gpu_options`<a id="gpu_options"></a>
</td>
<td>
`GPUOptions gpu_options`
</td>
</tr><tr>
<td>
`graph_options`<a id="graph_options"></a>
</td>
<td>
`GraphOptions graph_options`
</td>
</tr><tr>
<td>
`inter_op_parallelism_threads`<a id="inter_op_parallelism_threads"></a>
</td>
<td>
`int32 inter_op_parallelism_threads`
</td>
</tr><tr>
<td>
`intra_op_parallelism_threads`<a id="intra_op_parallelism_threads"></a>
</td>
<td>
`int32 intra_op_parallelism_threads`
</td>
</tr><tr>
<td>
`isolate_session_state`<a id="isolate_session_state"></a>
</td>
<td>
`bool isolate_session_state`
</td>
</tr><tr>
<td>
`log_device_placement`<a id="log_device_placement"></a>
</td>
<td>
`bool log_device_placement`
</td>
</tr><tr>
<td>
`operation_timeout_in_ms`<a id="operation_timeout_in_ms"></a>
</td>
<td>
`int64 operation_timeout_in_ms`
</td>
</tr><tr>
<td>
`placement_period`<a id="placement_period"></a>
</td>
<td>
`int32 placement_period`
</td>
</tr><tr>
<td>
`rpc_options`<a id="rpc_options"></a>
</td>
<td>
`RPCOptions rpc_options`
</td>
</tr><tr>
<td>
`session_inter_op_thread_pool`<a id="session_inter_op_thread_pool"></a>
</td>
<td>
`repeated ThreadPoolOptionProto session_inter_op_thread_pool`
</td>
</tr><tr>
<td>
`share_cluster_devices_in_session`<a id="share_cluster_devices_in_session"></a>
</td>
<td>
`bool share_cluster_devices_in_session`
</td>
</tr><tr>
<td>
`use_per_session_threads`<a id="use_per_session_threads"></a>
</td>
<td>
`bool use_per_session_threads`
</td>
</tr>
</table>



## Child Classes
[`class DeviceCountEntry`](../../../tf/compat/v1/ConfigProto/DeviceCountEntry.md)

[`class Experimental`](../../../tf/compat/v1/ConfigProto/Experimental.md)

