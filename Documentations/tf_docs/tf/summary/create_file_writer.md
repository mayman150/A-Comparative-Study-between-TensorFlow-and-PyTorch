description: Creates a summary file writer for the given log directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.create_file_writer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.create_file_writer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/summary_ops_v2.py">View source</a>



Creates a summary file writer for the given log directory.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.create_file_writer(
    logdir,
    max_queue=None,
    flush_millis=None,
    filename_suffix=None,
    name=None,
    experimental_trackable=False,
    experimental_mesh=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`logdir`<a id="logdir"></a>
</td>
<td>
a string specifying the directory in which to write an event file.
</td>
</tr><tr>
<td>
`max_queue`<a id="max_queue"></a>
</td>
<td>
the largest number of summaries to keep in a queue; will flush
once the queue gets bigger than this. Defaults to 10.
</td>
</tr><tr>
<td>
`flush_millis`<a id="flush_millis"></a>
</td>
<td>
the largest interval between flushes. Defaults to 120,000.
</td>
</tr><tr>
<td>
`filename_suffix`<a id="filename_suffix"></a>
</td>
<td>
optional suffix for the event file name. Defaults to `.v2`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
a name for the op that creates the writer.
</td>
</tr><tr>
<td>
`experimental_trackable`<a id="experimental_trackable"></a>
</td>
<td>
a boolean that controls whether the returned writer
will be a `TrackableResource`, which makes it compatible with SavedModel
when used as a <a href="../../tf/Module.md"><code>tf.Module</code></a> property.
</td>
</tr><tr>
<td>
`experimental_mesh`<a id="experimental_mesh"></a>
</td>
<td>
a <a href="../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a> instance. When running
with DTensor, the mesh (experimental_mesh.host_mesh()) will be used for
bringing all the DTensor logging from accelerator to CPU mesh.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A SummaryWriter object.
</td>
</tr>

</table>

