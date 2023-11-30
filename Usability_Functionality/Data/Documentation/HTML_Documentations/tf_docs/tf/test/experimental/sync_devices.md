description: Synchronizes all devices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.test.experimental.sync_devices" />
<meta itemprop="path" content="Stable" />
</div>

# tf.test.experimental.sync_devices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/test_util.py">View source</a>



Synchronizes all devices.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.test.experimental.sync_devices`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.test.experimental.sync_devices()
</code></pre>



<!-- Placeholder for "Used in" -->

By default, GPUs run asynchronously. This means that when you run an op on the
GPU, like <a href="../../../tf/linalg/matmul.md"><code>tf.linalg.matmul</code></a>, the op may still be running on the GPU when the
function returns. Non-GPU devices can also be made to run asynchronously by
calling <a href="../../../tf/config/experimental/set_synchronous_execution.md"><code>tf.config.experimental.set_synchronous_execution(False)</code></a>. Calling
`sync_devices()` blocks until pending ops have finished executing. This is
primarily useful for measuring performance during a benchmark.

For example, here is how you can measure how long <a href="../../../tf/linalg/matmul.md"><code>tf.linalg.matmul</code></a> runs:

```
>>> import time
>>> x = tf.random.normal((4096, 4096))
>>> tf.linalg.matmul(x, x)  # Warmup.
>>> tf.test.experimental.sync_devices()  # Block until warmup has completed.
>>>
>>> start = time.time()
>>> y = tf.linalg.matmul(x, x)
>>> tf.test.experimental.sync_devices()  # Block until matmul has completed.
>>> end = time.time()
>>> print(f'Time taken: {end - start}')
```

If the call to `sync_devices()` was omitted, the time printed could be too
small. This is because the op could still be running asynchronously when
the line `end = time.time()` is executed.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
If run outside Eager mode. This must be called in Eager mode,
outside any <a href="../../../tf/function.md"><code>tf.function</code></a>s.
</td>
</tr>
</table>

