<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.tpu" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.contrib.tpu

Ops related to Tensor Processing Units.

<!-- Placeholder for "Used in" -->










## Modules

[`profiler`](../../tf/contrib/tpu/profiler.md) module: Stub file to maintain backwards compatibility.

## Classes

[`class AsyncCheckpointSaverHook`](../../tf/contrib/tpu/AsyncCheckpointSaverHook.md): Saves checkpoints every N steps or seconds.

[`class CrossShardOptimizer`](../../tf/tpu/CrossShardOptimizer.md): An optimizer that averages gradients across TPU shards.

[`class DeviceAssignment`](../../tf/tpu/experimental/DeviceAssignment.md): Mapping from logical cores in a computation to the physical TPU topology.

[`class InfeedQueue`](../../tf/contrib/tpu/InfeedQueue.md): A helper object to build a device infeed queue.

[`class InputPipelineConfig`](../../tf/estimator/tpu/InputPipelineConfig.md): Please see the definition of these values in TPUConfig.

[`class RunConfig`](../../tf/estimator/tpu/RunConfig.md): RunConfig with TPU support.

[`class TPUConfig`](../../tf/estimator/tpu/TPUConfig.md): TPU related configuration required by `TPUEstimator`.

[`class TPUDistributionStrategy`](../../tf/contrib/tpu/TPUDistributionStrategy.md): The strategy to run Keras model on TPU.

[`class TPUEstimator`](../../tf/estimator/tpu/TPUEstimator.md): Estimator with TPU support.

[`class TPUEstimatorSpec`](../../tf/estimator/tpu/TPUEstimatorSpec.md): Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

[`class Topology`](../../tf/contrib/tpu/Topology.md): Describes a set of TPU devices.

## Functions

[`batch_parallel(...)`](../../tf/tpu/batch_parallel.md): Shards `computation` along the batch dimension for parallel execution.

[`bfloat16_scope(...)`](../../tf/tpu/bfloat16_scope.md): Scope class for bfloat16 variables so that the model uses custom getter.

[`core(...)`](../../tf/tpu/core.md): Returns the device name for a core in a replicated TPU computation.

[`cross_replica_sum(...)`](../../tf/tpu/cross_replica_sum.md): Sum the input tensor across replicas according to group_assignment.

[`device_assignment(...)`](../../tf/contrib/tpu/device_assignment.md): Computes a device_assignment of a computation across a TPU topology.

[`export_estimator_savedmodel(...)`](../../tf/contrib/tpu/export_estimator_savedmodel.md): Export `Estimator` trained model for TPU inference.

[`infeed_dequeue(...)`](../../tf/contrib/tpu/infeed_dequeue.md): A placeholder op for a value that will be fed into the computation.

[`infeed_dequeue_tuple(...)`](../../tf/contrib/tpu/infeed_dequeue_tuple.md): A placeholder op for values fed into the TPU simultaneously as a tuple.

[`infeed_enqueue(...)`](../../tf/contrib/tpu/infeed_enqueue.md): An op which feeds a single Tensor value into the computation.

[`infeed_enqueue_tuple(...)`](../../tf/contrib/tpu/infeed_enqueue_tuple.md): Feeds multiple Tensor values into the computation as an XLA tuple.

[`initialize_system(...)`](../../tf/tpu/initialize_system.md): Initializes a distributed TPU system for use with TensorFlow.

[`keras_to_tpu_model(...)`](../../tf/contrib/tpu/keras_to_tpu_model.md): Copy `model` along with weights to the TPU. (deprecated)

[`outfeed_dequeue(...)`](../../tf/contrib/tpu/outfeed_dequeue.md): Retrieves a single tensor from the computation outfeed.

[`outfeed_dequeue_tuple(...)`](../../tf/contrib/tpu/outfeed_dequeue_tuple.md): Retrieve multiple values from the computation outfeed.

[`outfeed_enqueue(...)`](../../tf/contrib/tpu/outfeed_enqueue.md): Enqueue a Tensor on the computation outfeed.

[`outfeed_enqueue_tuple(...)`](../../tf/contrib/tpu/outfeed_enqueue_tuple.md): Enqueue multiple Tensor values on the computation outfeed.

[`outside_compilation(...)`](../../tf/tpu/outside_compilation.md): Builds part of a computation outside any current TPU replicate scope.

[`repeat(...)`](../../tf/contrib/tpu/repeat.md): Builds a training loop that executes a fixed number of iterations.

[`replicate(...)`](../../tf/tpu/replicate.md): Builds a graph operator that runs a replicated TPU computation.

[`rewrite(...)`](../../tf/tpu/rewrite.md): Rewrites `computation` for execution on a TPU system.

[`shard(...)`](../../tf/tpu/shard.md): Shards `computation` for parallel execution.

[`shutdown_system(...)`](../../tf/tpu/shutdown_system.md): Shuts down a running a distributed TPU system.

[`while_loop(...)`](../../tf/contrib/tpu/while_loop.md): Builds a training loop for TPUs.

