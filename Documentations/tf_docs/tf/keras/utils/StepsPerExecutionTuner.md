description: Steps per execution tuner class.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.StepsPerExecutionTuner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="start"/>
<meta itemprop="property" content="stop"/>
</div>

# tf.keras.utils.StepsPerExecutionTuner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/steps_per_execution_tuning.py#L25-L264">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Steps per execution tuner class.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.StepsPerExecutionTuner(
    optimizer,
    spe_variable,
    interval=5,
    change_spe_interval=10,
    change_threshold=0.1
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`optimizer`<a id="optimizer"></a>
</td>
<td>
The optimizer used for training/evaluation/prediction. Used
to measure iterations and global throughput
(`optimizer.iterations`/second).
</td>
</tr><tr>
<td>
`spe_variable`<a id="spe_variable"></a>
</td>
<td>
A <a href="../../../tf/Variable.md"><code>tf.Variable</code></a> representing the `steps_per_execution`
variable used during training/evaluation/prediction. Must be
updatable with `spe_variable.assign`.
</td>
</tr><tr>
<td>
`interval`<a id="interval"></a>
</td>
<td>
Optional int, the amount of seconds to wait between calls to
measure throughput and tune `spe_variable`. Defaults to 5.
</td>
</tr><tr>
<td>
`change_spe_interval`<a id="change_spe_interval"></a>
</td>
<td>
Optional int, the number of throughput measurements
before tuning. Defaults to 10.
</td>
</tr><tr>
<td>
`change_threshold`<a id="change_threshold"></a>
</td>
<td>
Optional float, the percent different in throughput to
trigger a `steps_per_execution` change. For example, `0.1` triggers
changes if throughput changes more than 10%.
</td>
</tr>
</table>



#### Examples:



If you're using `model.compile` and `model.fit`, this functionality is
available at compile time with `steps_per_execution='auto'`

```python
model.compile(..., steps_per_execution='auto')
```

Custom training loop usage:

```python
# Get model
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Create our steps per execution variable
steps_per_execution = tf.Variable(
    1,
    dtype="int64",
    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
)

# Create the tuner
tuner = StepsPerExecutionTuner(
    optimizer, steps_per_execution
)

# Create a step function that runs a single training step
@tf.function
def step_fn(iterator):
    batch_data, labels = next(iterator)
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        loss_value = loss_fn(labels, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

# We can now pack multiple execution steps into one call
@tf.function
def multi_step_train_fn(iterator, steps_per_execution):
    for _ in tf.range(steps_per_execution):
        outputs = step_fn(iterator)
    return

initial_steps_per_execution = 1
steps_per_epoch = 100
epochs = 2

# Start the tuner before training
tuner.start()

# We can now call our multi step training with our data
for epoch in range(epochs):
    for _ in range(steps_per_epoch):
        multi_step_train_fn(iterator, steps_per_execution)

# End the tuner after training
tuner.stop()
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`steps_per_execution`<a id="steps_per_execution"></a>
</td>
<td>
Settable attribute representing`steps_per_execution` variable.
</td>
</tr>
</table>



## Methods

<h3 id="start"><code>start</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/steps_per_execution_tuning.py#L136-L149">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start()
</code></pre>

Starts steps per execution tuning thread.

Returns a `threading.Thread` which will run every `self.interval`
    seconds to measure throughput and tune steps per execution.

<h3 id="stop"><code>stop</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/steps_per_execution_tuning.py#L180-L183">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stop()
</code></pre>

Stops steps per execution tuning thread.




