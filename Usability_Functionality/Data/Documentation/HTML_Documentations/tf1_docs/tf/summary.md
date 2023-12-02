<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.summary

Operations for writing summary data, for use in analysis and visualization.

<!-- Placeholder for "Used in" -->

See the [Summaries and
TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) guide.

## Classes

[`class Event`](../tf/Event.md): A ProtocolMessage

[`class FileWriter`](../tf/summary/FileWriter.md): Writes `Summary` protocol buffers to event files.

[`class FileWriterCache`](../tf/summary/FileWriterCache.md): Cache for file writers.

[`class SessionLog`](../tf/SessionLog.md): A ProtocolMessage

[`class Summary`](../tf/Summary.md): A ProtocolMessage

[`class SummaryDescription`](../tf/summary/SummaryDescription.md): A ProtocolMessage

[`class TaggedRunMetadata`](../tf/summary/TaggedRunMetadata.md): A ProtocolMessage

## Functions

[`all_v2_summary_ops(...)`](../tf/summary/all_v2_summary_ops.md): Returns all V2-style summary ops defined in the current default graph.

[`audio(...)`](../tf/summary/audio.md): Outputs a `Summary` protocol buffer with audio.

[`get_summary_description(...)`](../tf/summary/get_summary_description.md): Given a TensorSummary node_def, retrieve its SummaryDescription.

[`histogram(...)`](../tf/summary/histogram.md): Outputs a `Summary` protocol buffer with a histogram.

[`image(...)`](../tf/summary/image.md): Outputs a `Summary` protocol buffer with images.

[`initialize(...)`](../tf/summary/initialize.md): Initializes summary writing for graph execution mode.

[`merge(...)`](../tf/summary/merge.md): Merges summaries.

[`merge_all(...)`](../tf/summary/merge_all.md): Merges all summaries collected in the default graph.

[`scalar(...)`](../tf/summary/scalar.md): Outputs a `Summary` protocol buffer containing a single scalar value.

[`tensor_summary(...)`](../tf/summary/tensor_summary.md): Outputs a `Summary` protocol buffer with a serialized tensor.proto.

[`text(...)`](../tf/summary/text.md): Summarizes textual data.

