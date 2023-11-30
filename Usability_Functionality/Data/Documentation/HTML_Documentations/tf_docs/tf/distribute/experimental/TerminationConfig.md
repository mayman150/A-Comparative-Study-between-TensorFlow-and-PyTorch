description: Customization of PreemptionCheckpointHandler for various platforms.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.TerminationConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.distribute.experimental.TerminationConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/failure_handling.py">View source</a>



Customization of `PreemptionCheckpointHandler` for various platforms.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.distribute.experimental.TerminationConfig(
    termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

A `TerminationConfig` can be created and passed to a
<a href="../../../tf/distribute/experimental/PreemptionCheckpointHandler.md"><code>tf.distribute.experimental.PreemptionCheckpointHandler</code></a> to provide
customization based on the platform. It can deliver three pieces of
information:

* How to decide if there is a termination event soon

The form of termination notification and how to fetch it vary across
platforms. Thus `PreemptionCheckpointHandler` may take a user-defined
function, `termination_watcher_fn`, and execute it repeatedly to check for
termination notification. `termination_watcher_fn` should be a function
that returns `True` if a termination notification is available and
`False` otherwise. The function should be lightweight and non-blocking so that
resources can be cleaned up properly if no termination signal is ever raised
until training finishes.

* How to exit the program

A user can configure this through the `exit_fn`, which
`PreemptionCheckpointHandler` executes after saving the checkpoint to exit the
training program gracefully. For <a href="../../../tf/distribute/MultiWorkerMirroredStrategy.md"><code>tf.distribute.MultiWorkerMirroredStrategy</code></a>,
a restart is necessary to reset the program's state. However, having a
customized `exit_fn` may facilitate the restart and smoothen the training
experience. How so? Maybe the platform has an agreement to a `RESTART_CODE`
recognized as a program auto-restart signal, or maybe the user has a
coordinating script that starts up the training, in which they can configure
the program to auto-restart if it ever exits with this `RESTART_CODE`. In both
cases, configuring the `exit_fn` to be `sys.exit(RESTART_CODE)` makes the
training seamless.

* How long does `PreemptionCheckpointHandler` have from receiving a
termination event notice till the actual termination

Some platforms have a gap time as long as one hour or so. In these cases,
there is the option to utilize this gap time for training as much as possible
before saving a checkpoint and exiting. This can be achieved by passing the
`grace_period` argument a nonzero value. Note, for a user with a grace period
that is not multiple times longer than their checkpoint writing time (e.g.,
three times or more), we advise not to configure this argument, in which case
`PreemptionCheckpointHandler` will directly save a checkpoint and exit.


**The default behavior**:

* For Google Borg Platform:
    * Automatically know how to detect preemption signal
    * Exit with a platform-recognized restart code
    * Save a checkpoint and exit immediately

* For Google Cloud Platform:
    * Automatically know how to detect maintenance signal.
    * Exit with a code (User may configure this)
    * Automatically utilized the extended training period before save and exit

* For Other platform:
    * If `termination_watcher_fn` is `None`, we will treat `signal.SIGTERM` as
    a termination signal.
    * If `exit_fn` is not configured, we exit the program with an arbitrary
    code.
    * If `grace_period` is not configured, we will wrap up the current
    training step, save a checkpoint, and exit the program as soon as we
    receive the termination signal.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`termination_watcher_fn`<a id="termination_watcher_fn"></a>
</td>
<td>
a function to execute repeatedly that returns
`True` if a preemption signal is available and False otherwise. The
function cannot block until a preemption signal is available, which
prevents proper cleanup of the program. A change is **NOT** recommended
for users on Google Borg or Google Cloud Platform.
</td>
</tr><tr>
<td>
`exit_fn`<a id="exit_fn"></a>
</td>
<td>
a function to execute after a checkpoint is saved and before the
preemption happens. Usually, it should be in the form of
`lambda: sys.exit(RESTART_CODE)`, where `RESTART_CODE` varies by
platform. A change is **NOT** recommended for users on Google Borg.
Users on Google Cloud Platform may configure it to use a customized
`RESTART_CODE`.
</td>
</tr><tr>
<td>
`grace_period`<a id="grace_period"></a>
</td>
<td>
the length of time between receiving a preemption signal and
the actual preemption. A change is **NOT** recommended for users on
Google Borg, Google Cloud Platform, or users with a short grace period.
</td>
</tr><tr>
<td>
`save_fn`<a id="save_fn"></a>
</td>
<td>
an optional function letting you configure how to save a
checkpoint. This is useful if you'd like to pass extra argument to
<a href="../../../tf/train/CheckpointManager.md#save"><code>tf.train.CheckpointManager.save</code></a> or <a href="../../../tf/train/Checkpoint.md#save"><code>tf.train.Checkpoint.save</code></a>. By
default, if not configured, the API will save checkpoint without extra
arguments.
</td>
</tr>
</table>



