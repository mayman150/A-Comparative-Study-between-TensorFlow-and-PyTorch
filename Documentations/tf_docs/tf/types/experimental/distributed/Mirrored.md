description: Holds a distributed value: a map from replica id to synchronized values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.distributed.Mirrored" />
<meta itemprop="path" content="Stable" />
</div>

# tf.types.experimental.distributed.Mirrored

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/distribute.py">View source</a>



Holds a distributed value: a map from replica id to synchronized values.

Inherits From: [`DistributedValues`](../../../../tf/distribute/DistributedValues.md)

<!-- Placeholder for "Used in" -->

`Mirrored` values are <a href="../../../../tf/distribute/DistributedValues.md"><code>tf.distribute.DistributedValues</code></a> for which we know that
the value on all replicas is the same. `Mirrored` values are kept synchronized
by the distribution strategy in use, while `tf.types.experimental.PerReplica`
values are left unsynchronized. `Mirrored` values typically represent model
weights. We can safely read a `Mirrored` value in a cross-replica context by
using the value on any replica, while `PerReplica` values should not be read
or manipulated directly by the user in a cross-replica context.

