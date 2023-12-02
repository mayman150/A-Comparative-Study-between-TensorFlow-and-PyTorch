<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.checkpoint" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.contrib.checkpoint

Tools for working with object-based checkpoints.

<!-- Placeholder for "Used in" -->

Visualization and inspection:

#### Managing dependencies:



Trackable data structures:

#### Checkpoint management:



Saving and restoring Python state:

## Classes

[`class CheckpointManager`](../../tf/train/CheckpointManager.md): Deletes old checkpoints.

[`class Checkpointable`](../../tf/contrib/checkpoint/Checkpointable.md): Manages dependencies on other objects.

[`class CheckpointableBase`](../../tf/contrib/checkpoint/CheckpointableBase.md): Base class for `Trackable` objects without automatic dependencies.

[`class CheckpointableObjectGraph`](../../tf/contrib/checkpoint/CheckpointableObjectGraph.md): A ProtocolMessage

[`class List`](../../tf/contrib/checkpoint/List.md): An append-only sequence type which is trackable.

[`class Mapping`](../../tf/contrib/checkpoint/Mapping.md): An append-only trackable mapping data structure with string keys.

[`class NoDependency`](../../tf/contrib/checkpoint/NoDependency.md): Allows attribute assignment to `Trackable` objects with no dependency.

[`class NumpyState`](../../tf/contrib/checkpoint/NumpyState.md): A trackable object whose NumPy array attributes are saved/restored.

[`class PythonStateWrapper`](../../tf/train/experimental/PythonState.md): A mixin for putting Python state in an object-based checkpoint.

[`class UniqueNameTracker`](../../tf/contrib/checkpoint/UniqueNameTracker.md): Adds dependencies on trackable objects with name hints.

## Functions

[`capture_dependencies(...)`](../../tf/contrib/checkpoint/capture_dependencies.md): Capture variables created within this scope as `Template` dependencies.

[`dot_graph_from_checkpoint(...)`](../../tf/contrib/checkpoint/dot_graph_from_checkpoint.md): Visualizes an object-based checkpoint (from <a href="../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>).

[`list_objects(...)`](../../tf/contrib/checkpoint/list_objects.md): Traverse the object graph and list all accessible objects.

[`object_metadata(...)`](../../tf/contrib/checkpoint/object_metadata.md): Retrieves information about the objects in a checkpoint.

[`split_dependency(...)`](../../tf/contrib/checkpoint/split_dependency.md): Creates multiple dependencies with a synchronized save/restore.

