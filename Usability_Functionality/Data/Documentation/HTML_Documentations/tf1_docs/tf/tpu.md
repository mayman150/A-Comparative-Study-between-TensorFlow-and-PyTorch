<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.tpu

Ops related to Tensor Processing Units.

<!-- Placeholder for "Used in" -->


## Modules

[`experimental`](../tf/tpu/experimental.md) module: Public API for tf.tpu.experimental namespace.

## Classes

[`class CrossShardOptimizer`](../tf/tpu/CrossShardOptimizer.md): An optimizer that averages gradients across TPU shards.

## Functions

[`batch_parallel(...)`](../tf/tpu/batch_parallel.md): Shards `computation` along the batch dimension for parallel execution.

[`bfloat16_scope(...)`](../tf/tpu/bfloat16_scope.md): Scope class for bfloat16 variables so that the model uses custom getter.

[`core(...)`](../tf/tpu/core.md): Returns the device name for a core in a replicated TPU computation.

[`cross_replica_sum(...)`](../tf/tpu/cross_replica_sum.md): Sum the input tensor across replicas according to group_assignment.

[`initialize_system(...)`](../tf/tpu/initialize_system.md): Initializes a distributed TPU system for use with TensorFlow.

[`outside_compilation(...)`](../tf/tpu/outside_compilation.md): Builds part of a computation outside any current TPU replicate scope.

[`replicate(...)`](../tf/tpu/replicate.md): Builds a graph operator that runs a replicated TPU computation.

[`rewrite(...)`](../tf/tpu/rewrite.md): Rewrites `computation` for execution on a TPU system.

[`shard(...)`](../tf/tpu/shard.md): Shards `computation` for parallel execution.

[`shutdown_system(...)`](../tf/tpu/shutdown_system.md): Shuts down a running a distributed TPU system.

