description: Stop training when a monitored metric has stopped improving.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.EarlyStopping" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_monitor_value"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.EarlyStopping

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L1980-L2153">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Stop training when a monitored metric has stopped improving.

Inherits From: [`Callback`](../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.EarlyStopping(
    monitor=&#x27;val_loss&#x27;,
    min_delta=0,
    patience=0,
    verbose=0,
    mode=&#x27;auto&#x27;,
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

Assuming the goal of a training is to minimize the loss. With this, the
metric to be monitored would be `'loss'`, and mode would be `'min'`. A
`model.fit()` training loop will check at end of every epoch whether
the loss is no longer decreasing, considering the `min_delta` and
`patience` if applicable. Once it's found no longer decreasing,
`model.stop_training` is marked True and the training terminates.

The quantity to be monitored needs to be available in `logs` dict.
To make it so, pass the loss or metrics at `model.compile()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`monitor`<a id="monitor"></a>
</td>
<td>
Quantity to be monitored.
</td>
</tr><tr>
<td>
`min_delta`<a id="min_delta"></a>
</td>
<td>
Minimum change in the monitored quantity
to qualify as an improvement, i.e. an absolute
change of less than min_delta, will count as no
improvement.
</td>
</tr><tr>
<td>
`patience`<a id="patience"></a>
</td>
<td>
Number of epochs with no improvement
after which training will be stopped.
</td>
</tr><tr>
<td>
`verbose`<a id="verbose"></a>
</td>
<td>
Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
displays messages when the callback takes an action.
</td>
</tr><tr>
<td>
`mode`<a id="mode"></a>
</td>
<td>
One of `{"auto", "min", "max"}`. In `min` mode,
training will stop when the quantity
monitored has stopped decreasing; in `"max"`
mode it will stop when the quantity
monitored has stopped increasing; in `"auto"`
mode, the direction is automatically inferred
from the name of the monitored quantity.
</td>
</tr><tr>
<td>
`baseline`<a id="baseline"></a>
</td>
<td>
Baseline value for the monitored quantity.
Training will stop if the model doesn't show improvement over the
baseline.
</td>
</tr><tr>
<td>
`restore_best_weights`<a id="restore_best_weights"></a>
</td>
<td>
Whether to restore model weights from
the epoch with the best value of the monitored quantity.
If False, the model weights obtained at the last step of
training are used. An epoch will be restored regardless
of the performance relative to the `baseline`. If no epoch
improves on `baseline`, training will run for `patience`
epochs and restore weights from the best epoch in that set.
</td>
</tr><tr>
<td>
`start_from_epoch`<a id="start_from_epoch"></a>
</td>
<td>
Number of epochs to wait before starting
to monitor improvement. This allows for a warm-up period in which
no improvement is expected and thus training will not be stopped.
</td>
</tr>
</table>



#### Example:



```
>>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
>>> # This callback will stop the training when there is no improvement in
>>> # the loss for three consecutive epochs.
>>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
>>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=10, batch_size=1, callbacks=[callback],
...                     verbose=0)
>>> len(history.history['loss'])  # Only 4 epochs are run.
4
```

## Methods

<h3 id="get_monitor_value"><code>get_monitor_value</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L2140-L2150">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_monitor_value(
    logs
)
</code></pre>




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






