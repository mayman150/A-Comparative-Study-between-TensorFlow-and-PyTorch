description: Enable visualizations for TensorBoard.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.TensorBoard" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.TensorBoard

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L2348-L3004">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Enable visualizations for TensorBoard.

Inherits From: [`Callback`](../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.TensorBoard(
    log_dir=&#x27;logs&#x27;,
    histogram_freq=0,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq=&#x27;epoch&#x27;,
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

TensorBoard is a visualization tool provided with TensorFlow.

This callback logs events for TensorBoard, including:

* Metrics summary plots
* Training graph visualization
* Weight histograms
* Sampled profiling

When used in <a href="../../../tf/keras/Model.md#evaluate"><code>Model.evaluate</code></a> or regular validation
([on_test_end](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback#on_test_end)),
in addition to epoch summaries, there will be a summary that records
evaluation metrics vs `Model.optimizer.iterations` written. The metric names
will be prepended with `evaluation`, with `Model.optimizer.iterations` being
the step in the visualized TensorBoard.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:

```
tensorboard --logdir=path_to_your_logs
```

You can find more information about TensorBoard
[here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`log_dir`<a id="log_dir"></a>
</td>
<td>
the path of the directory where to save the log files to be
parsed by TensorBoard. e.g. log_dir = os.path.join(working_dir,
'logs') This directory should not be reused by any other callbacks.
</td>
</tr><tr>
<td>
`histogram_freq`<a id="histogram_freq"></a>
</td>
<td>
frequency (in epochs) at which to compute
weight histograms for the layers of the model. If set to 0, histograms
won't be computed. Validation data (or split) must be specified for
histogram visualizations.
</td>
</tr><tr>
<td>
`write_graph`<a id="write_graph"></a>
</td>
<td>
whether to visualize the graph in TensorBoard. The log file
can become quite large when write_graph is set to True.
</td>
</tr><tr>
<td>
`write_images`<a id="write_images"></a>
</td>
<td>
whether to write model weights to visualize as image in
TensorBoard.
</td>
</tr><tr>
<td>
`write_steps_per_second`<a id="write_steps_per_second"></a>
</td>
<td>
whether to log the training steps per second
into TensorBoard. This supports both epoch and batch frequency
logging.
</td>
</tr><tr>
<td>
`update_freq`<a id="update_freq"></a>
</td>
<td>
`'batch'` or `'epoch'` or integer. When using `'epoch'`,
writes the losses and metrics to TensorBoard after every epoch.
If using an integer, let's say `1000`, all metrics and losses
(including custom ones added by <a href="../../../tf/keras/Model.md#compile"><code>Model.compile</code></a>) will be logged to
TensorBoard every 1000 batches. `'batch'` is a synonym for `1`,
meaning that they will be written every batch.
Note however that writing too frequently to TensorBoard can slow down
your training, especially when used with <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> as
it will incur additional synchronization overhead.
Use with `ParameterServerStrategy` is not supported.
Batch-level summary writing is also available via `train_step`
override. Please see
[TensorBoard Scalars tutorial](https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)  # noqa: E501
for more details.
</td>
</tr><tr>
<td>
`profile_batch`<a id="profile_batch"></a>
</td>
<td>
Profile the batch(es) to sample compute characteristics.
profile_batch must be a non-negative integer or a tuple of integers.
A pair of positive integers signify a range of batches to profile.
By default, profiling is disabled.
</td>
</tr><tr>
<td>
`embeddings_freq`<a id="embeddings_freq"></a>
</td>
<td>
frequency (in epochs) at which embedding layers will be
visualized. If set to 0, embeddings won't be visualized.
</td>
</tr><tr>
<td>
`embeddings_metadata`<a id="embeddings_metadata"></a>
</td>
<td>
Dictionary which maps embedding layer names to the
filename of a file in which to save metadata for the embedding layer.
In case the same metadata file is to be
used for all embedding layers, a single filename can be passed.
</td>
</tr>
</table>



#### Examples:




#### Basic usage:



```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
# Then run the tensorboard command to view the visualizations.
```

Custom batch-level summaries in a subclassed Model:

```python
class MyModel(tf.keras.Model):

  def build(self, _):
    self.dense = tf.keras.layers.Dense(10)

  def call(self, x):
    outputs = self.dense(x)
    tf.summary.histogram('outputs', outputs)
    return outputs

model = MyModel()
model.compile('sgd', 'mse')

# Make sure to set `update_freq=N` to log a batch-level summary every N
# batches.  In addition to any `tf.summary` contained in `Model.call`,
# metrics added in `Model.compile` will be logged every N batches.
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
model.fit(x_train, y_train, callbacks=[tb_callback])
```

Custom batch-level summaries in a Functional API Model:

```python
def my_summary(x):
  tf.summary.histogram('x', x)
  return x

inputs = tf.keras.Input(10)
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Lambda(my_summary)(x)
model = tf.keras.Model(inputs, outputs)
model.compile('sgd', 'mse')

# Make sure to set `update_freq=N` to log a batch-level summary every N
# batches. In addition to any `tf.summary` contained in `Model.call`,
# metrics added in `Model.compile` will be logged every N batches.
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
model.fit(x_train, y_train, callbacks=[tb_callback])
```

#### Profiling:



```python
# Profile a single batch, e.g. the 5th batch.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', profile_batch=5)
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])

# Profile a range of batches, e.g. from 10 to 20.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', profile_batch=(10,20))
model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
```

## Methods

<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L2567-L2585">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>

Sets Keras model and writes graph if specified.


<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L691-L692">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






