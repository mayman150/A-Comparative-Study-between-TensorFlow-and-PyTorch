description: Callback to save the best Keras model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.SidecarEvaluatorModelExport" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.SidecarEvaluatorModelExport

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/sidecar_evaluator.py#L341-L432">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Callback to save the best Keras model.

Inherits From: [`ModelCheckpoint`](../../../tf/keras/callbacks/ModelCheckpoint.md), [`Callback`](../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.SidecarEvaluatorModelExport(
    export_filepath, checkpoint_filepath, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

It expands the functionality of the existing ModelCheckpoint callback to
enable exporting the best models after evaluation with validation dataset.

When using the `SidecarEvaluatorModelExport` callback in conjunction with
<a href="../../../tf/keras/utils/SidecarEvaluator.md"><code>keras.utils.SidecarEvaluator</code></a>, users should provide the `filepath`, which
is the path for this callback to export model or save weights to, and
`ckpt_filepath`, which is where the checkpoint is available to extract
the epoch number from. The callback will then export the model that the
evaluator deems as the best (among the checkpoints saved by the training
counterpart) to the specified `filepath`. This callback is intended to be
used by SidecarEvaluator only.

#### Example:



```python
model.compile(loss=..., optimizer=...,
              metrics=['accuracy'])
sidecar_evaluator = keras.utils.SidecarEvaluator(
    model=model,
    data=dataset,
    checkpoint_dir=checkpoint_dir,
    max_evaluations=1,
    callbacks=[
        SidecarEvaluatorModelExport(
            export_filepath=os.path.join(checkpoint_dir,
                                  'best_model_eval',
                                  'best-model-{epoch:04d}'),
            checkpoint_filepath=os.path.join(checkpoint_dir,
            'ckpt-{epoch:04d}'),
            save_freq="eval",
            save_weights_only=True,
            monitor="loss",
            mode="min",
            verbose=1,
        ),
    ],
)
sidecar_evaluator.start()
# Model weights are saved if evaluator deems it's the best seen so far.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`export_filepath`<a id="export_filepath"></a>
</td>
<td>
Path where best models should be saved by this
`SidecarEvaluatorModelExport` callback. Epoch formatting options, such
as `os.path.join(best_model_dir, 'best-model-{epoch:04d}')`, can be
used to allow saved model to preserve epoch information in the file
name. SidecarEvaluatorModelExport will use the "training epoch" at
which the checkpoint was saved by training to fill the epoch
placeholder in the path.
</td>
</tr><tr>
<td>
`checkpoint_filepath`<a id="checkpoint_filepath"></a>
</td>
<td>
Path where checkpoints were saved by training. This
should be the same as what is provided to `filepath` argument of
`ModelCheckpoint` on the training side, such as
`os.path.join(checkpoint_dir, 'ckpt-{epoch:04d}')`.
</td>
</tr>
</table>



## Methods

<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L694-L695">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>




<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L691-L692">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






