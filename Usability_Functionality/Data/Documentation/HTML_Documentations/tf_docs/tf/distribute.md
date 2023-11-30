description: Public API for tf._api.v2.distribute namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.distribute

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.distribute namespace



## Modules

[`cluster_resolver`](../tf/distribute/cluster_resolver.md) module: Public API for tf._api.v2.distribute.cluster_resolver namespace

[`coordinator`](../tf/distribute/coordinator.md) module: Public API for tf._api.v2.distribute.coordinator namespace

[`experimental`](../tf/distribute/experimental.md) module: Public API for tf._api.v2.distribute.experimental namespace

## Classes

[`class CrossDeviceOps`](../tf/distribute/CrossDeviceOps.md): Base class for cross-device reduction and broadcasting algorithms.

[`class DistributedDataset`](../tf/distribute/DistributedDataset.md): Represents a dataset distributed among devices and machines.

[`class DistributedIterator`](../tf/distribute/DistributedIterator.md): An iterator over <a href="../tf/distribute/DistributedDataset.md"><code>tf.distribute.DistributedDataset</code></a>.

[`class DistributedValues`](../tf/distribute/DistributedValues.md): Base class for representing distributed values.

[`class HierarchicalCopyAllReduce`](../tf/distribute/HierarchicalCopyAllReduce.md): Hierarchical copy all-reduce implementation of CrossDeviceOps.

[`class InputContext`](../tf/distribute/InputContext.md): A class wrapping information needed by an input function.

[`class InputOptions`](../tf/distribute/InputOptions.md): Run options for `experimental_distribute_dataset(s_from_function)`.

[`class InputReplicationMode`](../tf/distribute/InputReplicationMode.md): Replication mode for input function.

[`class MirroredStrategy`](../tf/distribute/MirroredStrategy.md): Synchronous training across multiple replicas on one machine.

[`class MultiWorkerMirroredStrategy`](../tf/distribute/MultiWorkerMirroredStrategy.md): A distribution strategy for synchronous training on multiple workers.

[`class NcclAllReduce`](../tf/distribute/NcclAllReduce.md): NCCL all-reduce implementation of CrossDeviceOps.

[`class OneDeviceStrategy`](../tf/distribute/OneDeviceStrategy.md): A distribution strategy for running on a single device.

[`class ParameterServerStrategy`](../tf/distribute/experimental/ParameterServerStrategy.md): An multi-worker tf.distribute strategy with parameter servers.

[`class ReduceOp`](../tf/distribute/ReduceOp.md): Indicates how a set of values should be reduced.

[`class ReductionToOneDevice`](../tf/distribute/ReductionToOneDevice.md): A CrossDeviceOps implementation that copies values to one device to reduce.

[`class ReplicaContext`](../tf/distribute/ReplicaContext.md): A class with a collection of APIs that can be called in a replica context.

[`class RunOptions`](../tf/distribute/RunOptions.md): Run options for `strategy.run`.

[`class Server`](../tf/distribute/Server.md): An in-process TensorFlow server, for use in distributed training.

[`class Strategy`](../tf/distribute/Strategy.md): A state & compute distribution policy on a list of devices.

[`class StrategyExtended`](../tf/distribute/StrategyExtended.md): Additional APIs for algorithms that need to be distribution-aware.

[`class TPUStrategy`](../tf/distribute/TPUStrategy.md): Synchronous training on TPUs and TPU Pods.

## Functions

[`experimental_set_strategy(...)`](../tf/distribute/experimental_set_strategy.md): Set a <a href="../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> as current without `with strategy.scope()`.

[`get_replica_context(...)`](../tf/distribute/get_replica_context.md): Returns the current <a href="../tf/distribute/ReplicaContext.md"><code>tf.distribute.ReplicaContext</code></a> or `None`.

[`get_strategy(...)`](../tf/distribute/get_strategy.md): Returns the current <a href="../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> object.

[`has_strategy(...)`](../tf/distribute/has_strategy.md): Return if there is a current non-default <a href="../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.

[`in_cross_replica_context(...)`](../tf/distribute/in_cross_replica_context.md): Returns `True` if in a cross-replica context.

