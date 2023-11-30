description: Deprecated. Please use <a href="../../../tf/keras/utils/SidecarEvaluator.md"><code>tf.keras.utils.SidecarEvaluator</code></a> instead.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.experimental.SidecarEvaluator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="start"/>
</div>

# tf.keras.experimental.SidecarEvaluator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/sidecar_evaluator.py#L322-L338">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Deprecated. Please use <a href="../../../tf/keras/utils/SidecarEvaluator.md"><code>tf.keras.utils.SidecarEvaluator</code></a> instead.

Inherits From: [`SidecarEvaluator`](../../../tf/keras/utils/SidecarEvaluator.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.experimental.SidecarEvaluator(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Caution: <a href="../../../tf/keras/experimental/SidecarEvaluator.md"><code>tf.keras.experimental.SidecarEvaluator</code></a> endpoint is
  deprecated and will be removed in a future release. Please use
  <a href="../../../tf/keras/utils/SidecarEvaluator.md"><code>tf.keras.utils.SidecarEvaluator</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`<a id="model"></a>
</td>
<td>
Model to use for evaluation. The model object used here should
be a <a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a>, and should be the same as the one that is
used in training, where <a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a>s are checkpointed. The
model should have one or more metrics compiled before using
`SidecarEvaluator`.
</td>
</tr><tr>
<td>
`data`<a id="data"></a>
</td>
<td>
The input data for evaluation. `SidecarEvaluator` supports all
data types that Keras `model.evaluate` supports as the input data
`x`, such as a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>.
</td>
</tr><tr>
<td>
`checkpoint_dir`<a id="checkpoint_dir"></a>
</td>
<td>
Directory where checkpoint files are saved.
</td>
</tr><tr>
<td>
`steps`<a id="steps"></a>
</td>
<td>
Number of steps to perform evaluation for, when evaluating a
single checkpoint file. If `None`, evaluation continues until the
dataset is exhausted. For repeated evaluation dataset, user must
specify `steps` to avoid infinite evaluation loop.
</td>
</tr><tr>
<td>
`max_evaluations`<a id="max_evaluations"></a>
</td>
<td>
Maximum number of the checkpoint file to be
evaluated, for `SidecarEvaluator` to know when to stop. The
evaluator will stop after it evaluates a checkpoint filepath ending
with '<ckpt_name>-<max_evaluations>'. If using
<a href="../../../tf/train/CheckpointManager.md#save"><code>tf.train.CheckpointManager.save</code></a> for saving checkpoints, the kth
saved checkpoint has the filepath suffix '<ckpt_name>-<k>' (k=1 for
the first saved), and if checkpoints are saved every epoch after
training, the filepath saved at the kth epoch would end with
'<ckpt_name>-<k>. Thus, if training runs for n epochs, and the
evaluator should end after the training finishes, use n for this
parameter. Note that this is not necessarily equal to the number of
total evaluations, since some checkpoints may be skipped if
evaluation is slower than checkpoint creation. If `None`,
`SidecarEvaluator` will evaluate indefinitely, and the user must
terminate evaluator program themselves.
</td>
</tr><tr>
<td>
`callbacks`<a id="callbacks"></a>
</td>
<td>
List of <a href="../../../tf/keras/callbacks/Callback.md"><code>keras.callbacks.Callback</code></a> instances to apply
during evaluation. See
[callbacks](/api_docs/python/tf/keras/callbacks).
</td>
</tr>
</table>



## Methods

<h3 id="start"><code>start</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/sidecar_evaluator.py#L206-L319">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start()
</code></pre>

Starts the evaluation loop.




