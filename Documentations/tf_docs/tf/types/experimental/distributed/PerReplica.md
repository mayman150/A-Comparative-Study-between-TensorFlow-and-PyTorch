description: Holds a distributed value: a map from replica id to unsynchronized values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.distributed.PerReplica" />
<meta itemprop="path" content="Stable" />
</div>

# tf.types.experimental.distributed.PerReplica

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/distribute.py">View source</a>



Holds a distributed value: a map from replica id to unsynchronized values.

Inherits From: [`DistributedValues`](../../../../tf/distribute/DistributedValues.md)

<!-- Placeholder for "Used in" -->

`PerReplica` values exist on the worker devices, with a different value for
each replica. They can be produced many ways, often by iterating through a
distributed dataset returned by
<a href="../../../../tf/distribute/Strategy.md#experimental_distribute_dataset"><code>tf.distribute.Strategy.experimental_distribute_dataset</code></a> and
<a href="../../../../tf/distribute/Strategy.md#distribute_datasets_from_function"><code>tf.distribute.Strategy.distribute_datasets_from_function</code></a>. They are also the
typical result returned by <a href="../../../../tf/distribute/Strategy.md#run"><code>tf.distribute.Strategy.run</code></a>.

