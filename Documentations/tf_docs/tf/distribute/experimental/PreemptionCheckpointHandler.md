description: Preemption and error handler for synchronous training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.PreemptionCheckpointHandler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
<meta itemprop="property" content="save_checkpoint_if_preempted"/>
<meta itemprop="property" content="watch_preemption_scope"/>
</div>

# tf.distribute.experimental.PreemptionCheckpointHandler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/failure_handling.py">View source</a>



Preemption and error handler for synchronous training.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.distribute.experimental.PreemptionCheckpointHandler(
    cluster_resolver,
    checkpoint_or_checkpoint_manager,
    checkpoint_dir=None,
    termination_config=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: This API only supports use with
<a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a> and <a href="../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>.

A `PreemptionCheckpointHandler` coordinates all workers to save a checkpoint
upon receiving a preemption signal. It also helps disseminate application
error messages accurately among the cluster. When a
`PreemptionCheckpointHandler` object is created, it restores values from
the latest checkpoint file if any exists.

Right after the initialization, the object starts to watch out for termination
signal for any member in the cluster. If receiving a signal, the next time the
worker executes <a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a>, the
`PreemptionCheckpointHandler` will align all workers to save a checkpoint.
Then, if an `exit_fn` is configured via
<a href="../../../tf/distribute/experimental/TerminationConfig.md"><code>tf.distribute.experimental.TerminationConfig</code></a>, it will be invoked. Otherwise,
the process will simply exit and later the platform should restart it.

Note: We advise users of <a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a> who
choose to configure their
own `exit_fn` in <a href="../../../tf/distribute/experimental/TerminationConfig.md"><code>tf.distribute.experimental.TerminationConfig</code></a> to include a
`sys.exit(CODE_OR_MESSAGE)` in the `exit_fn` so that after the restart, all
workers can initialize communication services correctly. For users of
<a href="../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>, if they do not wish to do a cluster restart but
would like an in-process restart (i.e., keep the coordinator alive and re-do
the steps to connect to cluster, initialize TPU system, and make the
`TPUStrategy` object), they could configure the `exit_fn` to a no-op.

For users of <a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a>, the core API is
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a>:

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()

trained_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='step_in_epoch')

with strategy.scope():
  dataset, model, optimizer = ...

  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                   model=model,
                                   trained_epoch=trained_epoch,
                                   step_in_epoch=step_in_epoch)

  preemption_checkpoint_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint, checkpoint_dir)

while trained_epoch.numpy() < NUM_EPOCH:

  while step_in_epoch.numpy() < STEPS_PER_EPOCH:

    # distributed_train_function contains a call to strategy.run.
    loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
    # For users of MultiWorkerMirroredStrategy, usually
    # STEPS_PER_TRAIN_FUNCTION = 1.
    step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
    ...

  epoch.assign_add(1)
  step_in_epoch.assign(0)
```

For users of <a href="../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>, the core APIs are
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a> and
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#watch_preemption_scope"><code>PreemptionCheckpointHandler.watch_preemption_scope</code></a>:

```python

strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)

# Rest of TPU init omitted, see documentation for TPUSTrategy.

with preemption_checkpoint_handler.watch_preemption_scope():
  while trained_epoch.numpy() < NUM_EPOCH:

    while step_in_epoch.numpy() < STEPS_PER_EPOCH:

      # distributed_train_function contains a call to strategy.run.
      loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))

      # For users of TPUStrategy, usually STEPS_PER_TRAIN_FUNCTION >> 1 since
      # clustering multiple steps within a tf.function amortizes the overhead
      # of launching a multi-device function on TPU Pod.
      step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
      ...

    epoch.assign_add(1)
    step_in_epoch.assign(0)
```

Not all interruptions come with advance notice so that the
`PreemptionCheckpointHandler` can handle them, e.g., those caused by hardware
failure. For a user who saves checkpoints for these cases themselves outside
the `PreemptionCheckpointHandler`, if they are using a
<a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a>, pass it as the
`checkpoint_or_checkpoint_manager` argument to the
`PreemptionCheckpointHandler`. If they do not have a
<a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a> but are directly working with
<a href="../../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>, we advise saving the checkpoints in the directory
that's passed as the `checkpoint_dir` argument. In this way, at the program
beginning, `PreemptionCheckpointHandler` can restore the latest checkpoint
from the directory, no matter it's saved by the user themselves or saved by
the `PreemptionCheckpointHandler` before preemption happens.

**A note on the platform:**

`PreemptionCheckpointHandler` can only handle the kind of termination with
advance notice. For now, the API recognizes the termination signal for CPU,
GPU, and TPU on Google Borg and CPU and GPU on the Google Cloud Platform. In
these cases, `PreemptionCheckpointHandler` will automatically adopt the
correct preemption/maintenance notification detection mechanism. Users of
other platforms can configure a detection monitoring behavior through the
<a href="../../../tf/distribute/experimental/TerminationConfig.md"><code>tf.distribute.experimental.TerminationConfig</code></a>. Customization for the exit
behavior and grace period length could also be done here.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cluster_resolver`<a id="cluster_resolver"></a>
</td>
<td>
a <a href="../../../tf/distribute/cluster_resolver/ClusterResolver.md"><code>tf.distribute.cluster_resolver.ClusterResolver</code></a>
object. You may also obtain it through the `cluster_resolver` attribute
of the distribution strategy in use.
</td>
</tr><tr>
<td>
`checkpoint_or_checkpoint_manager`<a id="checkpoint_or_checkpoint_manager"></a>
</td>
<td>
a <a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a> or a
<a href="../../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a>. If you are using a <a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a>
to manage checkpoints outside the `PreemptionCheckpointHandler` for
backup purpose as well, pass it as `checkpoint_or_checkpoint_manager`
argument. Otherwise, pass a <a href="../../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a> and the
`PreemptionCheckpointHandler` will create
a <a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a> to manage it in the `checkpoint_dir`.
</td>
</tr><tr>
<td>
`checkpoint_dir`<a id="checkpoint_dir"></a>
</td>
<td>
a directory where the `PreemptionCheckpointHandler` saves
and restores checkpoints. When a `PreemptionCheckpointHandler` is
created, the latest checkpoint in the `checkpoint_dir` will be restored.
(This is not needed if a <a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a> instead of a
<a href="../../../tf/train/Checkpoint.md"><code>tf.train.Checkpoint</code></a> is passed as the
`checkpoint_or_checkpoint_manager` argument.)
</td>
</tr><tr>
<td>
`termination_config`<a id="termination_config"></a>
</td>
<td>
optional, a
<a href="../../../tf/distribute/experimental/TerminationConfig.md"><code>tf.distribute.experimental.TerminationConfig</code></a> object to configure for a
platform other than Google Borg or GCP.
</td>
</tr>
</table>



## Methods

<h3 id="run"><code>run</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/failure_handling.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    distributed_train_function, *args, **kwargs
)
</code></pre>

Runs a training function with error and preemption handling.

This function handles the preemption signal from any peer in the cluster by
saving the training progress and exiting gracefully. It will
also broadcase any program error encountered during the execution of
`distributed_train_function` to all workers so that they can raise the same
error.

The `distributed_train_function` argument should be a distributed train
function (i.e., containing a call to <a href="../../../tf/distribute/Strategy.md#run"><code>tf.distribute.Strategy.run</code></a>). For
<a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a> users, we recommend passing in a
single-step `distributed_train_function` to
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a> so that the checkpoint can be saved in
time in case a preemption signal or maintenance notice is sent.

Besides the preemption and error handling part,
`PreemptionCheckpointHandler.run(distributed_train_function, *args,
**kwargs)` has the same effect and output as
`distributed_train_function(*args, **kwargs)`. `distributed_train_function`
can return either some or no result. The following is a shortened example:

```python

@tf.function
def distributed_train_step(iterator):
  # A distributed single-step training function.

  def step_fn(inputs):
    # A per-replica single-step training function.
    x, y = inputs
    ...
    return loss

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

for epoch in range(preemption_handler.total_run_calls // STEPS_PER_EPOCH,
                   EPOCHS_TO_RUN):
  iterator = iter(multi_worker_dataset)
  total_loss = 0.0
  num_batches = 0

  for step in range(preemption_handler.total_run_calls % STEPS_PER_EPOCH,
                    STEPS_PER_EPOCH):
    total_loss += preemption_handler.run(distributed_train_step)
    num_batches += 1

  train_loss = total_loss / num_batches
  print('Epoch: %d, train_loss: %f.' %(epoch.numpy(), train_loss))

  train_accuracy.reset_states()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`distributed_train_function`
</td>
<td>
A (single-step) distributed training function.
</td>
</tr><tr>
<td>
`*args`
</td>
<td>
args for `distributed_train_function`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
kwargs for `distributed_train_function`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
Program error encountered by any member in the cluster while executing the
`distributed_train_function`, or any error from the program error
propagation process.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Result of running the `distributed_train_function`.
</td>
</tr>

</table>



<h3 id="save_checkpoint_if_preempted"><code>save_checkpoint_if_preempted</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/failure_handling.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_checkpoint_if_preempted(
    *args, **kwargs
)
</code></pre>

Saves a checkpoint if a preemption signal has been made available.

This is an alternative API for <a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a> and
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#watch_preemption_scope"><code>PreemptionCheckpointHandler.watch_preemption_scope</code></a>. This method works for
both <a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a> and
<a href="../../../tf/distribute/TPUStrategy.md"><code>tf.distribute.TPUStrategy</code></a>. However, **for TPUStrategy, this method will
add a synchronization point between workers and the coordinator** and thus
may have performance implication. If this is a concern, use the combination
of <a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#watch_preemption_scope"><code>PreemptionCheckpointHandler.watch_preemption_scope</code></a> and
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a> instead.

```python
strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
# initialization omitted

with strategy.scope():
  # Save in the checkpoint.
  trained_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='trained_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

  checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory, max_to_keep=1)
  preemption_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint_manager)

while trained_step.numpy() < NUM_STEPS:
  # Train STEPS_IN_FUNCTION steps at once.
  train_multi_step_function()
  trained_step.assign_add(STEPS_IN_FUNCTION)
  preemption_handler.save_checkpoint_if_preempted()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
args for <a href="../../../tf/train/CheckpointManager.md#save"><code>tf.train.CheckpointManager.save()</code></a> to save checkpoint.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
kwargs for <a href="../../../tf/train/CheckpointManager.md#save"><code>tf.train.CheckpointManager.save()</code></a> to save.
</td>
</tr>
</table>



<h3 id="watch_preemption_scope"><code>watch_preemption_scope</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/failure_handling.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@tf_contextlib.contextmanager</code>
<code>watch_preemption_scope()
</code></pre>

Syncs error and maybe save checkpoint for usage with TPUStrategy.

Note: Usage with <a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a> does not need
this API.

#### Example usage:



```python
with preemption_checkpoint_handler.watch_preemption_scope():
  while trained_step.numpy() < NUM_STEPS:

    # distributed_train_function contains a call to strategy.run.
    loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
    trained_step.assign_add(STEPS_PER_TRAIN_FUNCTION)
```

In this workflow, <a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md#run"><code>PreemptionCheckpointHandler.run</code></a> will flag preemption
signal received, and `watch_preemption_scope` will handle the preemption
signal by saving a checkpoint and then either exit to restart or execute a
user-passed `exit_fn` in <a href="../../../tf/distribute/experimental/TerminationConfig.md"><code>tf.distribute.experimental.TerminationConfig</code></a>. If
no preemption signal is received during execution of ops and function inside
the scope, `watch_preemption_scope` ensures the completion of all async op
and function execution when exiting and will raises exceptions if async
execution results in an error state.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Yields</th></tr>
<tr class="alt">
<td colspan="2">
None
</td>
</tr>

</table>





