<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.distribute" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.contrib.distribute

A distributed computation library for TF.

<!-- Placeholder for "Used in" -->

See [tensorflow/contrib/distribute/README.md](
https://www.tensorflow.org/code/tensorflow/contrib/distribute/README.md)
for overview and examples.

## Classes

[`class AllReduceCrossDeviceOps`](../../tf/contrib/distribute/AllReduceCrossDeviceOps.md): Reduction using all-reduce.

[`class CollectiveAllReduceStrategy`](../../tf/contrib/distribute/CollectiveAllReduceStrategy.md): Distribution strategy that uses collective ops for all-reduce.

[`class CrossDeviceOps`](../../tf/distribute/CrossDeviceOps.md): Base class for cross-device reduction and broadcasting algorithms.

[`class DistributeConfig`](../../tf/contrib/distribute/DistributeConfig.md): A config tuple for distribution strategies.

[`class DistributionStrategy`](../../tf/distribute/Strategy.md): A list of devices with a state & compute distribution policy.

[`class MirroredStrategy`](../../tf/contrib/distribute/MirroredStrategy.md): Mirrors vars to distribute across multiple devices and machines.

[`class Monitor`](../../tf/contrib/distribute/Monitor.md): Executes training steps, recovers and checkpoints.

[`class MultiWorkerAllReduce`](../../tf/contrib/distribute/MultiWorkerAllReduce.md): All-reduce algorithms for distributed TensorFlow.

[`class OneDeviceStrategy`](../../tf/distribute/OneDeviceStrategy.md): A distribution strategy for running on a single device.

[`class ParameterServerStrategy`](../../tf/contrib/distribute/ParameterServerStrategy.md): A parameter server DistributionStrategy.

[`class ReplicaContext`](../../tf/distribute/ReplicaContext.md): <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> API when in a replica context.

[`class StandardInputStep`](../../tf/contrib/distribute/StandardInputStep.md): Step with a standard implementation of input handling.

[`class StandardSingleLossStep`](../../tf/contrib/distribute/StandardSingleLossStep.md): A step function that implements a training step for a feed forward network.

[`class Step`](../../tf/contrib/distribute/Step.md): Interface for performing each step of a training algorithm.

[`class TPUStrategy`](../../tf/distribute/experimental/TPUStrategy.md): TPU distribution strategy implementation.

[`class UpdateContext`](../../tf/contrib/distribute/UpdateContext.md): Context manager when you are in `update()` or `update_non_slot()`.

## Functions

[`get_cross_replica_context(...)`](../../tf/contrib/distribute/get_cross_replica_context.md): Returns the current tf.distribute.Strategy if in a cross-replica context.

[`get_distribution_strategy(...)`](../../tf/distribute/get_strategy.md): Returns the current <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> object.

[`get_loss_reduction(...)`](../../tf/distribute/get_loss_reduction.md): <a href="../../tf/distribute/ReduceOp.md"><code>tf.distribute.ReduceOp</code></a> corresponding to the last loss reduction.

[`get_replica_context(...)`](../../tf/distribute/get_replica_context.md): Returns the current <a href="../../tf/distribute/ReplicaContext.md"><code>tf.distribute.ReplicaContext</code></a> or `None`.

[`get_strategy(...)`](../../tf/distribute/get_strategy.md): Returns the current <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> object.

[`has_distribution_strategy(...)`](../../tf/distribute/has_strategy.md): Return if there is a current non-default <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.

[`has_strategy(...)`](../../tf/distribute/has_strategy.md): Return if there is a current non-default <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>.

[`in_cross_replica_context(...)`](../../tf/distribute/in_cross_replica_context.md): Returns `True` if in a cross-replica context.

[`initialize_tpu_system(...)`](../../tf/tpu/experimental/initialize_tpu_system.md): Initialize the TPU devices.

[`require_replica_context(...)`](../../tf/contrib/distribute/require_replica_context.md): Verify in `replica_ctx` replica context.

[`run_standard_tensorflow_server(...)`](../../tf/contrib/distribute/run_standard_tensorflow_server.md): Starts a standard TensorFlow server.

