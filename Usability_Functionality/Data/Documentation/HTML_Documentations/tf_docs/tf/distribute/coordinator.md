description: Public API for tf._api.v2.distribute.coordinator namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.coordinator" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.distribute.coordinator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.distribute.coordinator namespace



## Classes

[`class ClusterCoordinator`](../../tf/distribute/experimental/coordinator/ClusterCoordinator.md): An object to schedule and coordinate remote function execution.

[`class PerWorkerValue`](../../tf/distribute/experimental/coordinator/PerWorkerValues.md): A container that holds a list of values, one value per worker.

[`class RemoteValue`](../../tf/distribute/experimental/coordinator/RemoteValue.md): An asynchronously available value of a scheduled function.

## Functions

[`experimental_get_current_worker_index(...)`](../../tf/distribute/coordinator/experimental_get_current_worker_index.md): Returns the current worker index, when called within a worker closure.

