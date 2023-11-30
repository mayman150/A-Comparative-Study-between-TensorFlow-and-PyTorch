description: Initializes accelerators and communication fabrics for DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.initialize_accelerator_system" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.initialize_accelerator_system

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/accelerator_util.py">View source</a>



Initializes accelerators and communication fabrics for DTensor.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.experimental.dtensor.initialize_multi_client`, `tf.experimental.dtensor.initialize_tpu_system`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.initialize_accelerator_system(
    device_type: Optional[str] = None,
    enable_coordination_service: Optional[bool] = True,
    num_logical_cpu_devices: Optional[int] = None,
    experimental_reset_context: Optional[bool] = False
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->

DTensor configures TensorFlow to run in the local mode or multi-client mode.
- In local mode, a mesh can only use devices attached to the current process.
- In multi-client mode, a mesh can span across devices from multiple clients.

If `DTENSOR_JOBS` is non-empty, DTensor configures TensorFlow to run in the
multi-client mode using the distributed runtime. In multi-client mode devices
on different clients can communicate with each other.

The following environment variables controls the behavior of this function.

- `DTENSOR_JOBS`: string, a comma separated list. Each item in the list is
    of format `{hostname}:{port}`. If empty, DTensor runs in the local mode.
    Examples of valid `DTENSOR_JOBS` values:
    - 4 clients on localhost:
      `localhost:10000,localhost:10001,localhost:10002,localhost:10003`
    - 2 clients on host1, 2 clients on host2
      `host1:10000,host1:10001,host2:10000,host2:10003`
    If the hostnames are BNS addresses, the items must be sorted in
    alphabetical order.
- `DTENSOR_CLIENT_ID`: integer, between `0` to `num_clients - 1`, to identify
    the client id of the current process. The default value is `0`.
- `DTENSOR_JOB_NAME`: string, a string for the name of the TensorFlow job.
    The job name controls the job name section of the TensorFlow DeviceSpecs,
    e.g., `job:worker` in `/job:worker/replica:0/task:0/device:TPU:0` when
    the job name is `worker`.
    The default value is `localhost` in local mode, and
    `worker` when in the multi-client mode. All DTensor clients within the
    same multi-client cluster share the same job name.
- `DTENSOR_USE_PARALLEL_EXECUTOR`: string, with its value being `pw` to
    specify that the backend is Pathways, and TensorFlow otherwise.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`device_type`<a id="device_type"></a>
</td>
<td>
Type of accelerator to use, can be CPU, GPU, or TPU. If None,
uses <a href="../../../tf/experimental/dtensor/preferred_device_type.md"><code>tf.experimental.dtensor.preferred_device_type()</code></a>.
</td>
</tr><tr>
<td>
`enable_coordination_service`<a id="enable_coordination_service"></a>
</td>
<td>
If true, enable distributed coordination
service to make sure that workers know the devices on each other, when
there is more than 1 client.
</td>
</tr><tr>
<td>
`num_logical_cpu_devices`<a id="num_logical_cpu_devices"></a>
</td>
<td>
the number of logical CPU devices per DTensor
client. Default to the current number of logical CPU
(`dtensor.num_local_devices("CPU")`),when `device_type` is CPU, otherwise
set automatially to match the number of local GPU/TPU devices.
</td>
</tr><tr>
<td>
`experimental_reset_context`<a id="experimental_reset_context"></a>
</td>
<td>
Reset the tensorflow context. Behaviors of
existing TensorFlow objects (e.g. Tensors) are undefined. Set this to True
as an escape hatch, if there is no clear way to refactor your code to call
initialize_accelerator_system() before calling TensorFlow APIs that
initialize the context.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`device_type`<a id="device_type"></a>
</td>
<td>
the type of accelerator that was initialized.
</td>
</tr>
</table>

