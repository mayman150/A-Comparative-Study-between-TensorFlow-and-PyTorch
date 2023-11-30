description: This class specifies the configurations for an Estimator run. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.estimator.RunConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="replace"/>
</div>

# tf.estimator.RunConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/run_config.py#L343-L957">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



This class specifies the configurations for an `Estimator` run. (deprecated)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.estimator.RunConfig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.estimator.RunConfig(
    model_dir=None,
    tf_random_seed=None,
    save_summary_steps=100,
    save_checkpoints_steps=_USE_DEFAULT,
    save_checkpoints_secs=_USE_DEFAULT,
    session_config=None,
    keep_checkpoint_max=5,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=100,
    train_distribute=None,
    device_fn=None,
    protocol=None,
    eval_distribute=None,
    experimental_distribute=None,
    experimental_max_worker_delay_secs=None,
    session_creation_timeout_secs=7200,
    checkpoint_save_graph_def=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.keras instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>
directory where model parameters, graph, etc are saved. If
`PathLike` object, the path will be resolved. If `None`, will use a
default value set by the Estimator.
</td>
</tr><tr>
<td>
`tf_random_seed`<a id="tf_random_seed"></a>
</td>
<td>
Random seed for TensorFlow initializers. Setting this
value allows consistency between reruns.
</td>
</tr><tr>
<td>
`save_summary_steps`<a id="save_summary_steps"></a>
</td>
<td>
Save summaries every this many steps.
</td>
</tr><tr>
<td>
`save_checkpoints_steps`<a id="save_checkpoints_steps"></a>
</td>
<td>
Save checkpoints every this many steps. Can not be
specified with `save_checkpoints_secs`.
</td>
</tr><tr>
<td>
`save_checkpoints_secs`<a id="save_checkpoints_secs"></a>
</td>
<td>
Save checkpoints every this many seconds. Can not
be specified with `save_checkpoints_steps`. Defaults to 600 seconds if
both `save_checkpoints_steps` and `save_checkpoints_secs` are not set in
constructor.  If both `save_checkpoints_steps` and
`save_checkpoints_secs` are `None`, then checkpoints are disabled.
</td>
</tr><tr>
<td>
`session_config`<a id="session_config"></a>
</td>
<td>
a ConfigProto used to set session parameters, or `None`.
</td>
</tr><tr>
<td>
`keep_checkpoint_max`<a id="keep_checkpoint_max"></a>
</td>
<td>
The maximum number of recent checkpoint files to
keep. As new files are created, older files are deleted. If `None` or 0,
all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
checkpoint files are kept). If a saver is passed to the estimator, this
argument will be ignored.
</td>
</tr><tr>
<td>
`keep_checkpoint_every_n_hours`<a id="keep_checkpoint_every_n_hours"></a>
</td>
<td>
Number of hours between each checkpoint to
be saved. The default value of 10,000 hours effectively disables the
feature.
</td>
</tr><tr>
<td>
`log_step_count_steps`<a id="log_step_count_steps"></a>
</td>
<td>
The frequency, in number of global steps, that the
global step and the loss will be logged during training.  Also controls
the frequency that the global steps / s will be logged (and written to
summary) during training.
</td>
</tr><tr>
<td>
`train_distribute`<a id="train_distribute"></a>
</td>
<td>
An optional instance of <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>. If
specified, then Estimator will distribute the user's model during
training, according to the policy specified by that strategy. Setting
`experimental_distribute.train_distribute` is preferred.
</td>
</tr><tr>
<td>
`device_fn`<a id="device_fn"></a>
</td>
<td>
A callable invoked for every `Operation` that takes the
`Operation` and returns the device string. If `None`, defaults to the
device function returned by `tf.train.replica_device_setter` with
round-robin strategy.
</td>
</tr><tr>
<td>
`protocol`<a id="protocol"></a>
</td>
<td>
An optional argument which specifies the protocol used when
starting server. `None` means default to grpc.
</td>
</tr><tr>
<td>
`eval_distribute`<a id="eval_distribute"></a>
</td>
<td>
An optional instance of <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>. If
specified, then Estimator will distribute the user's model during
evaluation, according to the policy specified by that strategy. Setting
`experimental_distribute.eval_distribute` is preferred.
</td>
</tr><tr>
<td>
`experimental_distribute`<a id="experimental_distribute"></a>
</td>
<td>
An optional
`tf.contrib.distribute.DistributeConfig` object specifying
DistributionStrategy-related configuration. The `train_distribute` and
`eval_distribute` can be passed as parameters to `RunConfig` or set in
`experimental_distribute` but not both.
</td>
</tr><tr>
<td>
`experimental_max_worker_delay_secs`<a id="experimental_max_worker_delay_secs"></a>
</td>
<td>
An optional integer specifying the
maximum time a worker should wait before starting. By default, workers
are started at staggered times, with each worker being delayed by up to
60 seconds. This is intended to reduce the risk of divergence, which can
occur when many workers simultaneously update the weights of a randomly
initialized model. Users who warm-start their models and train them for
short durations (a few minutes or less) should consider reducing this
default to improve training times.
</td>
</tr><tr>
<td>
`session_creation_timeout_secs`<a id="session_creation_timeout_secs"></a>
</td>
<td>
Max time workers should wait for a session
to become available (on initialization or when recovering a session)
with MonitoredTrainingSession. Defaults to 7200 seconds, but users may
want to set a lower value to detect problems with variable / session
(re)-initialization more quickly.
</td>
</tr><tr>
<td>
`checkpoint_save_graph_def`<a id="checkpoint_save_graph_def"></a>
</td>
<td>
Whether to save the GraphDef and MetaGraphDef
to `checkpoint_dir`. The GraphDef is saved after the session is created
as `graph.pbtxt`. MetaGraphDefs are saved out for every checkpoint as
`model.ckpt-*.meta`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If both `save_checkpoints_steps` and `save_checkpoints_secs`
are set.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`checkpoint_save_graph_def`<a id="checkpoint_save_graph_def"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`cluster_spec`<a id="cluster_spec"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`device_fn`<a id="device_fn"></a>
</td>
<td>
Returns the device_fn.

If device_fn is not `None`, it overrides the default
device function used in `Estimator`.
Otherwise the default one is used.
</td>
</tr><tr>
<td>
`eval_distribute`<a id="eval_distribute"></a>
</td>
<td>
Optional <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> for evaluation.
</td>
</tr><tr>
<td>
`evaluation_master`<a id="evaluation_master"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_max_worker_delay_secs`<a id="experimental_max_worker_delay_secs"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`global_id_in_cluster`<a id="global_id_in_cluster"></a>
</td>
<td>
The global id in the training cluster.

All global ids in the training cluster are assigned from an increasing
sequence of consecutive integers. The first id is 0.

Note: Task id (the property field `task_id`) is tracking the index of the
node among all nodes with the SAME task type. For example, given the cluster
definition as follows:

```
  cluster = {'chief': ['host0:2222'],
             'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
```

Nodes with task type `worker` can have id 0, 1, 2.  Nodes with task type
`ps` can have id, 0, 1. So, `task_id` is not unique, but the pair
(`task_type`, `task_id`) can uniquely determine a node in the cluster.

Global id, i.e., this field, is tracking the index of the node among ALL
nodes in the cluster. It is uniquely assigned.  For example, for the cluster
spec given above, the global ids are assigned as:
```
  task_type  | task_id  |  global_id
  --------------------------------
  chief      | 0        |  0
  worker     | 0        |  1
  worker     | 1        |  2
  worker     | 2        |  3
  ps         | 0        |  4
  ps         | 1        |  5
```
</td>
</tr><tr>
<td>
`is_chief`<a id="is_chief"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`keep_checkpoint_every_n_hours`<a id="keep_checkpoint_every_n_hours"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`keep_checkpoint_max`<a id="keep_checkpoint_max"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`log_step_count_steps`<a id="log_step_count_steps"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`master`<a id="master"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`model_dir`<a id="model_dir"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`num_ps_replicas`<a id="num_ps_replicas"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`num_worker_replicas`<a id="num_worker_replicas"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`protocol`<a id="protocol"></a>
</td>
<td>
Returns the optional protocol value.
</td>
</tr><tr>
<td>
`save_checkpoints_secs`<a id="save_checkpoints_secs"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`save_checkpoints_steps`<a id="save_checkpoints_steps"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`save_summary_steps`<a id="save_summary_steps"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`service`<a id="service"></a>
</td>
<td>
Returns the platform defined (in TF_CONFIG) service dict.
</td>
</tr><tr>
<td>
`session_config`<a id="session_config"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`session_creation_timeout_secs`<a id="session_creation_timeout_secs"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`task_id`<a id="task_id"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`task_type`<a id="task_type"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`tf_random_seed`<a id="tf_random_seed"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`train_distribute`<a id="train_distribute"></a>
</td>
<td>
Optional <a href="../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> for training.
</td>
</tr>
</table>



## Methods

<h3 id="replace"><code>replace</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/run_config.py#L883-L921">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>replace(
    **kwargs
)
</code></pre>

Returns a new instance of `RunConfig` replacing specified properties.

Only the properties in the following list are allowed to be replaced:

  - `model_dir`,
  - `tf_random_seed`,
  - `save_summary_steps`,
  - `save_checkpoints_steps`,
  - `save_checkpoints_secs`,
  - `session_config`,
  - `keep_checkpoint_max`,
  - `keep_checkpoint_every_n_hours`,
  - `log_step_count_steps`,
  - `train_distribute`,
  - `device_fn`,
  - `protocol`.
  - `eval_distribute`,
  - `experimental_distribute`,
  - `experimental_max_worker_delay_secs`,

In addition, either `save_checkpoints_steps` or `save_checkpoints_secs`
can be set (should not be both).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`**kwargs`
</td>
<td>
keyword named properties with new values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If any property name in `kwargs` does not exist or is not
allowed to be replaced, or both `save_checkpoints_steps` and
`save_checkpoints_secs` are set.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a new instance of `RunConfig`.
</td>
</tr>

</table>





