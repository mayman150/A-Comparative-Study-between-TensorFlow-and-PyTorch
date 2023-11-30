description: A ProtocolMessage

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.GPUOptions.Experimental" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="VirtualDevices"/>
</div>

# tf.compat.v1.GPUOptions.Experimental

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
`collective_ring_order`<a id="collective_ring_order"></a>
</td>
<td>
`string collective_ring_order`
</td>
</tr><tr>
<td>
`disallow_retry_on_allocation_failure`<a id="disallow_retry_on_allocation_failure"></a>
</td>
<td>
`bool disallow_retry_on_allocation_failure`
</td>
</tr><tr>
<td>
`gpu_host_mem_disallow_growth`<a id="gpu_host_mem_disallow_growth"></a>
</td>
<td>
`bool gpu_host_mem_disallow_growth`
</td>
</tr><tr>
<td>
`gpu_host_mem_limit_in_mb`<a id="gpu_host_mem_limit_in_mb"></a>
</td>
<td>
`float gpu_host_mem_limit_in_mb`
</td>
</tr><tr>
<td>
`gpu_system_memory_size_in_mb`<a id="gpu_system_memory_size_in_mb"></a>
</td>
<td>
`int32 gpu_system_memory_size_in_mb`
</td>
</tr><tr>
<td>
`internal_fragmentation_fraction`<a id="internal_fragmentation_fraction"></a>
</td>
<td>
`double internal_fragmentation_fraction`
</td>
</tr><tr>
<td>
`kernel_tracker_max_bytes`<a id="kernel_tracker_max_bytes"></a>
</td>
<td>
`int32 kernel_tracker_max_bytes`
</td>
</tr><tr>
<td>
`kernel_tracker_max_interval`<a id="kernel_tracker_max_interval"></a>
</td>
<td>
`int32 kernel_tracker_max_interval`
</td>
</tr><tr>
<td>
`kernel_tracker_max_pending`<a id="kernel_tracker_max_pending"></a>
</td>
<td>
`int32 kernel_tracker_max_pending`
</td>
</tr><tr>
<td>
`num_dev_to_dev_copy_streams`<a id="num_dev_to_dev_copy_streams"></a>
</td>
<td>
`int32 num_dev_to_dev_copy_streams`
</td>
</tr><tr>
<td>
`num_virtual_devices_per_gpu`<a id="num_virtual_devices_per_gpu"></a>
</td>
<td>
`int32 num_virtual_devices_per_gpu`
</td>
</tr><tr>
<td>
`timestamped_allocator`<a id="timestamped_allocator"></a>
</td>
<td>
`bool timestamped_allocator`
</td>
</tr><tr>
<td>
`use_cuda_malloc_async`<a id="use_cuda_malloc_async"></a>
</td>
<td>
`bool use_cuda_malloc_async`
</td>
</tr><tr>
<td>
`use_unified_memory`<a id="use_unified_memory"></a>
</td>
<td>
`bool use_unified_memory`
</td>
</tr><tr>
<td>
`virtual_devices`<a id="virtual_devices"></a>
</td>
<td>
`repeated VirtualDevices virtual_devices`
</td>
</tr>
</table>



## Child Classes
[`class VirtualDevices`](../../../../tf/compat/v1/GPUOptions/Experimental/VirtualDevices.md)

