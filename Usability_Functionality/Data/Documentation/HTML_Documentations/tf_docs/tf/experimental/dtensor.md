description: Public API for tf._api.v2.experimental.dtensor namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="MATCH"/>
<meta itemprop="property" content="UNSHARDED"/>
</div>

# Module: tf.experimental.dtensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.experimental.dtensor namespace



## Classes

[`class DTensorCheckpoint`](../../tf/experimental/dtensor/DTensorCheckpoint.md): Manages saving/restoring trackable values to disk, for DTensor. (deprecated)

[`class DTensorDataset`](../../tf/experimental/dtensor/DTensorDataset.md): A dataset of DTensors.

[`class DVariable`](../../tf/experimental/dtensor/DVariable.md): A replacement for tf.Variable which follows initial value placement.

[`class Layout`](../../tf/experimental/dtensor/Layout.md): Represents the layout information of a DTensor.

[`class Mesh`](../../tf/experimental/dtensor/Mesh.md): Represents a Mesh configuration over a certain list of Mesh Dimensions.

## Functions

[`barrier(...)`](../../tf/experimental/dtensor/barrier.md): Runs a barrier on the mesh.

[`call_with_layout(...)`](../../tf/experimental/dtensor/call_with_layout.md): Calls a function in the DTensor device scope if `layout` is not None.

[`check_layout(...)`](../../tf/experimental/dtensor/check_layout.md): Asserts that the layout of the DTensor is `layout`.

[`client_id(...)`](../../tf/experimental/dtensor/client_id.md): Returns this client's ID.

[`copy_to_mesh(...)`](../../tf/experimental/dtensor/copy_to_mesh.md): Copies a tf.Tensor onto the DTensor device with the given layout.

[`create_distributed_mesh(...)`](../../tf/experimental/dtensor/create_distributed_mesh.md): Creates a distributed mesh.

[`create_mesh(...)`](../../tf/experimental/dtensor/create_mesh.md): Creates a single-client mesh.

[`create_tpu_mesh(...)`](../../tf/experimental/dtensor/create_tpu_mesh.md): Returns a distributed TPU mesh optimized for AllReduce ring reductions.

[`default_mesh(...)`](../../tf/experimental/dtensor/default_mesh.md): Sets the default DTensor device mesh to use for enclosed functions.

[`device_name(...)`](../../tf/experimental/dtensor/device_name.md): Returns the singleton DTensor device's name.

[`enable_save_as_bf16(...)`](../../tf/experimental/dtensor/enable_save_as_bf16.md): Allows float32 DVariables to be checkpointed and restored as bfloat16.

[`fetch_layout(...)`](../../tf/experimental/dtensor/fetch_layout.md): Fetches the layout of a DTensor.

[`full_job_name(...)`](../../tf/experimental/dtensor/full_job_name.md): Returns the fully qualified TF job name for this or another task.

[`get_default_mesh(...)`](../../tf/experimental/dtensor/get_default_mesh.md): Return the default mesh under the current dtensor device context.

[`heartbeat_enabled(...)`](../../tf/experimental/dtensor/heartbeat_enabled.md): Returns true if DTensor heartbeat service is enabled.

[`initialize_accelerator_system(...)`](../../tf/experimental/dtensor/initialize_accelerator_system.md): Initializes accelerators and communication fabrics for DTensor.

[`initialize_multi_client(...)`](../../tf/experimental/dtensor/initialize_accelerator_system.md): Initializes accelerators and communication fabrics for DTensor.

[`initialize_tpu_system(...)`](../../tf/experimental/dtensor/initialize_accelerator_system.md): Initializes accelerators and communication fabrics for DTensor.

[`is_dtensor(...)`](../../tf/experimental/dtensor/is_dtensor.md): Check whether the input tensor is a DTensor.

[`job_name(...)`](../../tf/experimental/dtensor/job_name.md): Returns the job name used by all clients in this DTensor cluster.

[`jobs(...)`](../../tf/experimental/dtensor/jobs.md): Returns a list of job names of all clients in this DTensor cluster.

[`local_devices(...)`](../../tf/experimental/dtensor/local_devices.md): Returns a list of device specs configured on this client.

[`name_based_restore(...)`](../../tf/experimental/dtensor/name_based_restore.md): Restores from checkpoint_prefix to name based DTensors.

[`name_based_save(...)`](../../tf/experimental/dtensor/name_based_save.md): Saves name based Tensor into a Checkpoint.

[`num_clients(...)`](../../tf/experimental/dtensor/num_clients.md): Returns the number of clients in this DTensor cluster.

[`num_global_devices(...)`](../../tf/experimental/dtensor/num_global_devices.md): Returns the number of devices of device_type in this DTensor cluster.

[`num_local_devices(...)`](../../tf/experimental/dtensor/num_local_devices.md): Returns the number of devices of device_type configured on this client.

[`pack(...)`](../../tf/experimental/dtensor/pack.md): Packs <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> components into a DTensor.

[`preferred_device_type(...)`](../../tf/experimental/dtensor/preferred_device_type.md): Returns the preferred device type for the accelerators.

[`relayout(...)`](../../tf/experimental/dtensor/relayout.md): Changes the layout of `tensor`.

[`relayout_like(...)`](../../tf/experimental/dtensor/relayout_like.md): Changes the layout of `tensor` to the same as `layout_tensor`.

[`run_on(...)`](../../tf/experimental/dtensor/run_on.md): Runs enclosed functions in the DTensor device scope. (deprecated)

[`sharded_save(...)`](../../tf/experimental/dtensor/sharded_save.md): Saves given named tensor slices in a sharded, multi-client safe fashion.

[`shutdown_accelerator_system(...)`](../../tf/experimental/dtensor/shutdown_accelerator_system.md): Shuts down the accelerator system.

[`shutdown_tpu_system(...)`](../../tf/experimental/dtensor/shutdown_accelerator_system.md): Shuts down the accelerator system.

[`unpack(...)`](../../tf/experimental/dtensor/unpack.md): Unpacks a DTensor into <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> components.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
MATCH<a id="MATCH"></a>
</td>
<td>
`'match'`
</td>
</tr><tr>
<td>
UNSHARDED<a id="UNSHARDED"></a>
</td>
<td>
`'unsharded'`
</td>
</tr>
</table>

