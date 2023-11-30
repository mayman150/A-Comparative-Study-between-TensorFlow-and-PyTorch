description: Public API for tf._api.v2.train namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.train

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.train namespace



## Modules

[`experimental`](../tf/train/experimental.md) module: Public API for tf._api.v2.train.experimental namespace

## Classes

[`class BytesList`](../tf/train/BytesList.md): Used in <a href="../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Holds a list of byte-strings.

[`class Checkpoint`](../tf/train/Checkpoint.md): Manages saving/restoring trackable values to disk.

[`class CheckpointManager`](../tf/train/CheckpointManager.md): Manages multiple checkpoints by keeping some and deleting unneeded ones.

[`class CheckpointOptions`](../tf/train/CheckpointOptions.md): Options for constructing a Checkpoint.

[`class CheckpointView`](../tf/train/CheckpointView.md): Gathers and serializes a checkpoint view.

[`class ClusterDef`](../tf/train/ClusterDef.md): A ProtocolMessage

[`class ClusterSpec`](../tf/train/ClusterSpec.md): Represents a cluster as a set of "tasks", organized into "jobs".

[`class Coordinator`](../tf/train/Coordinator.md): A coordinator for threads.

[`class Example`](../tf/train/Example.md): An `Example` is a standard proto storing data for training and inference.

[`class ExponentialMovingAverage`](../tf/train/ExponentialMovingAverage.md): Maintains moving averages of variables by employing an exponential decay.

[`class Feature`](../tf/train/Feature.md): Used in <a href="../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Contains a list of values.

[`class FeatureList`](../tf/train/FeatureList.md): Mainly used as part of a <a href="../tf/train/SequenceExample.md"><code>tf.train.SequenceExample</code></a>.

[`class FeatureLists`](../tf/train/FeatureLists.md): Mainly used as part of a <a href="../tf/train/SequenceExample.md"><code>tf.train.SequenceExample</code></a>.

[`class Features`](../tf/train/Features.md): Used in <a href="../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Contains the mapping from keys to `Feature`.

[`class FloatList`](../tf/train/FloatList.md): Used in <a href="../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Holds a list of floats.

[`class Int64List`](../tf/train/Int64List.md): Used in <a href="../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Holds a list of Int64s.

[`class JobDef`](../tf/train/JobDef.md): A ProtocolMessage

[`class SequenceExample`](../tf/train/SequenceExample.md): A `SequenceExample` represents a sequence of features and some context.

[`class ServerDef`](../tf/train/ServerDef.md): A ProtocolMessage

[`class TrackableView`](../tf/train/TrackableView.md): Gathers and serializes a trackable view.

## Functions

[`checkpoints_iterator(...)`](../tf/train/checkpoints_iterator.md): Continuously yield new checkpoint files as they appear.

[`get_checkpoint_state(...)`](../tf/train/get_checkpoint_state.md): Returns CheckpointState proto from the "checkpoint" file.

[`latest_checkpoint(...)`](../tf/train/latest_checkpoint.md): Finds the filename of latest saved checkpoint file.

[`list_variables(...)`](../tf/train/list_variables.md): Lists the checkpoint keys and shapes of variables in a checkpoint.

[`load_checkpoint(...)`](../tf/train/load_checkpoint.md): Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

[`load_variable(...)`](../tf/train/load_variable.md): Returns the tensor value of the given variable in the checkpoint.

