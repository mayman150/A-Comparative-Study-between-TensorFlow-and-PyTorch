description: Options for constructing a Checkpoint.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.CheckpointOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.train.CheckpointOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint_options.py">View source</a>



Options for constructing a Checkpoint.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.CheckpointOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.CheckpointOptions(
    experimental_io_device=None,
    experimental_enable_async_checkpoint=False,
    experimental_write_callbacks=None,
    enable_async=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Used as the `options` argument to either <a href="../../tf/train/Checkpoint.md#save"><code>tf.train.Checkpoint.save()</code></a> or
<a href="../../tf/train/Checkpoint.md#restore"><code>tf.train.Checkpoint.restore()</code></a> methods to adjust how variables are
saved/restored.

Example: Run IO ops on "localhost" while saving a checkpoint:

```
step = tf.Variable(0, name="step")
checkpoint = tf.train.Checkpoint(step=step)
options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.save("/tmp/ckpt", options=options)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`experimental_io_device`<a id="experimental_io_device"></a>
</td>
<td>
string. Applies in a distributed setting.
Tensorflow device to use to access the filesystem. If `None` (default)
then for each variable the filesystem is accessed from the CPU:0 device
of the host where that variable is assigned. If specified, the
filesystem is instead accessed from that device for all variables.

This is for example useful if you want to save to a local directory,
such as "/tmp" when running in a distributed setting. In that case pass
a device for the host where the "/tmp" directory is accessible.
</td>
</tr><tr>
<td>
`experimental_enable_async_checkpoint`<a id="experimental_enable_async_checkpoint"></a>
</td>
<td>
bool Type. Deprecated, please use
the enable_async option.
</td>
</tr><tr>
<td>
`experimental_write_callbacks`<a id="experimental_write_callbacks"></a>
</td>
<td>
List[Callable]. A list of callback functions
that will be executed after each saving event finishes (i.e. after
`save()` or `write()`). For async checkpoint, the callbacks will be
executed only after the async thread finishes saving.

The return values of the callback(s) will be ignored. The callback(s)
can optionally take the `save_path` (the result of `save()` or
`write()`) as an argument. The callbacks will be executed in the same
order of this list after the checkpoint has been written.
</td>
</tr><tr>
<td>
`enable_async`<a id="enable_async"></a>
</td>
<td>
bool Type. Indicates whether async checkpointing is enabled.
Default is False, i.e., no async checkpoint.

Async checkpoint moves the checkpoint file writing off the main thread,
so that the model can continue to train while the checkpoing file
writing runs in the background. Async checkpoint reduces TPU device idle
cycles and speeds up model training process, while memory consumption
may increase.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`enable_async`<a id="enable_async"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_enable_async_checkpoint`<a id="experimental_enable_async_checkpoint"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_io_device`<a id="experimental_io_device"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_write_callbacks`<a id="experimental_write_callbacks"></a>
</td>
<td>

</td>
</tr>
</table>



