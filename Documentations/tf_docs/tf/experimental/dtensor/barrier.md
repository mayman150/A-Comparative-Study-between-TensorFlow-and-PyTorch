description: Runs a barrier on the mesh.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.barrier" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.barrier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/mesh_util.py">View source</a>



Runs a barrier on the mesh.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.barrier(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    barrier_name: Optional[str] = None,
    timeout_in_ms: Optional[int] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Upon returning from the barrier, all operations run before the barrier
would have completed across all clients. Currently we allocate a fully
sharded tensor with mesh shape and run an all_reduce on it.

#### Example:



A barrier can be used before application exit to ensure completion of pending
ops.

```python

x = [1, 2, 3]
x = dtensor.relayout(x, dtensor.Layout.batch_sharded(mesh, 'batch', 1))
dtensor.barrier(mesh)

# At this point all devices on all clients in the mesh have completed
# operations before the barrier. Therefore it is OK to tear down the clients.
sys.exit()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>
The mesh to run the barrier on.
</td>
</tr><tr>
<td>
`barrier_name`<a id="barrier_name"></a>
</td>
<td>
The name of the barrier. Mainly used for logging purpose.
</td>
</tr><tr>
<td>
`timeout_in_ms`<a id="timeout_in_ms"></a>
</td>
<td>
The timeout of the barrier in ms. If omitted, blocks
indefinitely till the barrier is reached from all clients.
</td>
</tr>
</table>

