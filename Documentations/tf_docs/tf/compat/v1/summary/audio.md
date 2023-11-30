description: Outputs a Summary protocol buffer with audio.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.summary.audio" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.summary.audio

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/summary/summary.py">View source</a>



Outputs a `Summary` protocol buffer with audio.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.summary.audio(
    name, tensor, sample_rate, max_outputs=3, collections=None, family=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

For compatibility purposes, when invoked in TF2 where the outermost context is
eager mode, this API will check if there is a suitable TF2 summary writer
context available, and if so will forward this call to that writer instead. A
"suitable" writer context means that the writer is set as the default writer,
and there is an associated non-empty value for `step` (see
<a href="../../../../tf/summary/SummaryWriter.md#as_default"><code>tf.summary.SummaryWriter.as_default</code></a>, `tf.summary.experimental.set_step` or
alternatively <a href="../../../../tf/compat/v1/train/create_global_step.md"><code>tf.compat.v1.train.create_global_step</code></a>). For the forwarded
call, the arguments here will be passed to the TF2 implementation of
<a href="../../../../tf/summary/audio.md"><code>tf.summary.audio</code></a>, and the return value will be an empty bytestring tensor,
to avoid duplicate summary writing. This forwarding is best-effort and not all
arguments will be preserved. Additionally:

* The TF2 op just outputs the data under a single tag that contains multiple
  samples, rather than multiple tags (i.e. no "/0" or "/1" suffixes).

To migrate to TF2, please use <a href="../../../../tf/summary/audio.md"><code>tf.summary.audio</code></a> instead. Please check
[Migrating tf.summary usage to
TF 2.0](https://www.tensorflow.org/tensorboard/migrate#in_tf_1x) for concrete
steps for migration.

#### How to Map Arguments

| TF1 Arg Name  | TF2 Arg Name    | Note                                   |
| :------------ | :-------------- | :------------------------------------- |
| `name`        | `name`          | -                                      |
| `tensor`      | `data`          | Input for this argument now must be    |
:               :                 : three-dimensional `[k, t, c]`, where   :
:               :                 : `k` is the number of audio clips, `t`  :
:               :                 : is the number of frames, and `c` is    :
:               :                 : the number of channels. Two-dimensional:
:               :                 : input is no longer supported.          :
| `sample_rate` | `sample_rate`   | -                                      |
| -             | `step`          | Explicit int64-castable monotonic step |
:               :                 : value. If omitted, this defaults to    :
:               :                 : `tf.summary.experimental.get_step()`.  :
| `max_outputs` | `max_outputs`   | -                                      |
| `collections` | Not Supported   | -                                      |
| `family`      | Removed         | Please use <a href="../../../../tf/name_scope.md"><code>tf.name_scope</code></a> instead to  |
:               :                 : manage summary name prefix.            :
| -             | `encoding`      | Optional constant str for the desired  |
:               :                 : encoding. Check the docs for           :
:               :                 : <a href="../../../../tf/summary/audio.md"><code>tf.summary.audio</code></a> for latest supported:
:               :                 : audio formats.                         :
| -             | `description`   | Optional long-form `str` description   |
:               :                 : for the summary. Markdown is supported.:
:               :                 : Defaults to empty.                     :


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
`sample_rate`.

The `tag` in the outputted Summary.Value protobufs is generated based on the
name, with a suffix depending on the max_outputs setting:

*  If `max_outputs` is 1, the summary value tag is '*name*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*name*/audio/0', '*name*/audio/1', etc

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the generated node. Will also serve as a series name in
TensorBoard.
</td>
</tr><tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
</td>
</tr><tr>
<td>
`sample_rate`<a id="sample_rate"></a>
</td>
<td>
A Scalar `float32` `Tensor` indicating the sample rate of the
signal in hertz.
</td>
</tr><tr>
<td>
`max_outputs`<a id="max_outputs"></a>
</td>
<td>
Max number of batch elements to generate audio for.
</td>
</tr><tr>
<td>
`collections`<a id="collections"></a>
</td>
<td>
Optional list of ops.GraphKeys.  The collections to add the
summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
</td>
</tr><tr>
<td>
`family`<a id="family"></a>
</td>
<td>
Optional; if provided, used as the prefix of the summary tag name,
which controls the tab name used for display on Tensorboard.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar `Tensor` of type `string`. The serialized `Summary` protocol
buffer.
</td>
</tr>

</table>


