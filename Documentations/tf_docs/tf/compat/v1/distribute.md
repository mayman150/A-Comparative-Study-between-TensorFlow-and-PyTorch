description: Public API for tf._api.v2.distribute namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.distribute" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.distribute

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.distribute namespace



## Modules

[`cluster_resolver`](../../../tf/compat/v1/distribute/cluster_resolver.md) module: Public API for tf._api.v2.distribute.cluster_resolver namespace

[`experimental`](../../../tf/compat/v1/distribute/experimental.md) module: Public API for tf._api.v2.distribute.experimental namespace

## Classes

[`class CrossDeviceOps`](../../../tf/distribute/CrossDeviceOps.md): Base class for cross-device reduction and broadcasting algorithms.

[`class HierarchicalCopyAllReduce`](../../../tf/distribute/HierarchicalCopyAllReduce.md): Hierarchical copy all-reduce implementation of CrossDeviceOps.

[`class InputContext`](../../../tf/distribute/InputContext.md): A class wrapping information needed by an input function.

[`class InputReplicationMode`](../../../tf/distribute/InputReplicationMode.md): Replication mode for input function.

[`class MirroredStrategy`](../../../tf/compat/v1/distribute/MirroredStrategy.md): Synchronous training across multiple replicas on one machine.

[`class NcclAllReduce`](../../../tf/distribute/NcclAllReduce.md): NCCL all-reduce implementation of CrossDeviceOps.

[`class OneDeviceStrategy`](../../../tf/compat/v1/distribute/OneDeviceStrategy.md): A distribution strategy for running on a single device.

[`class ReduceOp`](../../../tf/distribute/ReduceOp.md): Indicates how a set of values should be reduced.

[`class ReductionToOneDevice`](../../../tf/distribute/ReductionToOneDevice.md): A CrossDeviceOps implementation that copies values to one device to reduce.

[`class ReplicaContext`](../../../tf/compat/v1/distribute/ReplicaContext.md): A class with a collection of APIs that can be called in a replica context.

[`class RunOptions`](../../../tf/distribute/RunOptions.md): Run options for `strategy.run`.

[`class Server`](../../../tf/distribute/Server.md): An in-process TensorFlow server, for use in distributed training.

[`class Strategy`](../../../tf/compat/v1/distribute/Strategy.md): A list of devices with a state & compute distribution policy.

[`class StrategyExtended`](../../../tf/compat/v1/distribute/StrategyExtended.md): Additional APIs for algorithms that need to be distribution-aware.

## Functions

[`experimental_set_strategy(...)`](../../../tf/distribute/experimental_set_strategy.md): Set a <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> as current without `with strategy.scope()`.

[`get_loss_reduction(...)`](../../../tf/compat/v1/distribute/get_loss_reduction.md): <a href="../../../tf/distribute/ReduceOp.md"><code>tf.distribute.ReduceOp</code></a> corresponding to the last loss reduction.

[`get_replica_context(...)`](../../../tf/distribute/get_replica_context.md): Returns the current <a href="../../../tf/distribute/ReplicaContext.md"><code>tf.distribute.ReplicaContext</code></a> or `None`.

[`get_strategy(...)`](../../../tf/distribute/get_strategy.md): Returns the current <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> object.

[`has_strategy(...)`](../../../tf/distribute/has_strategy.md): Return if there is a current non-default <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.

[`in_cross_replica_context(...)`](../../../tf/distribute/in_cross_replica_context.md): Returns `True` if in a cross-replica context.

