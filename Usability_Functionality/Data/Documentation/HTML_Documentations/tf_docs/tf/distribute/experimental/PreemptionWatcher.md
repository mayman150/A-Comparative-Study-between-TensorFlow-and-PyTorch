description: Watch preemption signal and store it.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.experimental.PreemptionWatcher" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="block_until_worker_exit"/>
</div>

# tf.distribute.experimental.PreemptionWatcher

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/preemption_watcher.py">View source</a>



Watch preemption signal and store it.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.distribute.experimental.PreemptionWatcher()
</code></pre>



<!-- Placeholder for "Used in" -->

Notice: Currently only support Borg TPU environment with TPUClusterResolver.

This class provides a way to monitor the preemption signal during training on
TPU. It will start a background thread to watch the training process, trying
to fetch preemption message from the coordination service. When preemption
happens, the preempted worker will write the preemption message to the
coordination service. Thus getting a non-empty preemption message means there
is a preemption happened.

User can use the preemption message as a reliable preemption indicator, and
then set the coordinator to reconnect to the TPU worker instead of a fully
restart triggered by Borg. For example, a training process with
preemption recovery will be like:

```python
keep_running = True
preemption_watcher = None
while keep_running:
  try:
    # Initialize TPU cluster and stratygy.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    # PreemptionWatcher must be created after connected to cluster.
    preemption_watcher = tf.distribute.experimental.PreemptionWatcher()
    train_model(strategy)
    keep_running = False
  except Exception as e:
    if preemption_watcher and preemption_watcher.preemption_message:
      preemption_watcher.block_until_worker_exit()
      keep_running = True
    else:
      raise e
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`preemption_message`<a id="preemption_message"></a>
</td>
<td>
A variable to store the preemption message fetched from
the coordination service. If it is not None, then there is a preemption
happened.
</td>
</tr><tr>
<td>
`platform`<a id="platform"></a>
</td>
<td>
A PlatformDevice to indicate the current job's platform. Refer to
failure_handling_util.py for the definition of enum class PlatformDevice.
</td>
</tr>
</table>



## Methods

<h3 id="block_until_worker_exit"><code>block_until_worker_exit</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/failure_handling/preemption_watcher.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>block_until_worker_exit()
</code></pre>

Block coordinator until workers exit.

In some rare cases, another error could be raised during the
preemption grace period. This will cause the coordinator to reconnect to the
same TPU workers, which will be killed later. It prevents the coordinator to
reconnect to new TPU workers, and falls back to a hard restart. To avoid
this situation, this method will block the coordinator to reconnect until
workers exit. This method will be a no-op for non-TPU platform.



