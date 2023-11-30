description: Time-based interval Threads.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.TimedThread" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="is_alive"/>
<meta itemprop="property" content="on_interval"/>
<meta itemprop="property" content="start"/>
<meta itemprop="property" content="stop"/>
</div>

# tf.keras.utils.TimedThread

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L24-L148">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Time-based interval Threads.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.TimedThread(
    interval, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Runs a timed thread every x seconds. It can be used to run a threaded
function alongside model training or any other snippet of code.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`interval`<a id="interval"></a>
</td>
<td>
The interval, in seconds, to wait between calls to the
`on_interval` function.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
additional args that are passed to `threading.Thread`. By
default, `Thread` is started as a `daemon` thread unless
overridden by the user in `kwargs`.
</td>
</tr>
</table>



#### Examples:



```python
class TimedLogIterations(keras.utils.TimedThread):
    def __init__(self, model, interval):
        self.model = model
        super().__init__(interval)

    def on_interval(self):
        # Logs Optimizer iterations every x seconds
        try:
            opt_iterations = self.model.optimizer.iterations.numpy()
            print(f"Epoch: {epoch}, Optimizer Iterations: {opt_iterations}")
        except Exception as e:
            print(str(e))  # To prevent thread from getting killed

# `start` and `stop` the `TimerThread` manually. If the `on_interval` call
# requires access to `model` or other objects, override `__init__` method.
# Wrap it in a `try-except` to handle exceptions and `stop` the thread run.
timed_logs = TimedLogIterations(model=model, interval=5)
timed_logs.start()
try:
    model.fit(...)
finally:
    timed_logs.stop()

# Alternatively, run the `TimedThread` in a context manager
with TimedLogIterations(model=model, interval=5):
    model.fit(...)

# If the timed thread instance needs access to callback events,
# subclass both `TimedThread` and `Callback`.  Note that when calling
# `super`, they will have to called for each parent class if both of them
# have the method that needs to be run. Also, note that `Callback` has
# access to `model` as an attribute and need not be explictly provided.
class LogThreadCallback(
    keras.utils.TimedThread, keras.callbacks.Callback
):
    def __init__(self, interval):
        self._epoch = 0
        keras.utils.TimedThread.__init__(self, interval)
        keras.callbacks.Callback.__init__(self)

    def on_interval(self):
        if self.epoch:
            opt_iter = self.model.optimizer.iterations.numpy()
            logging.info(f"Epoch: {self._epoch}, Opt Iteration: {opt_iter}")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

with LogThreadCallback(interval=5) as thread_callback:
    # It's required to pass `thread_callback` to also `callbacks` arg of
    # `model.fit` to be triggered on callback events.
    model.fit(..., callbacks=[thread_callback])
```

## Methods

<h3 id="is_alive"><code>is_alive</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L127-L131">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_alive()
</code></pre>

Returns True if thread is running. Otherwise returns False.


<h3 id="on_interval"><code>on_interval</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L142-L148">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>on_interval()
</code></pre>

User-defined behavior that is called in the thread.


<h3 id="start"><code>start</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L109-L120">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start()
</code></pre>

Creates and starts the thread run.


<h3 id="stop"><code>stop</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L122-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stop()
</code></pre>

Stops the thread run.


<h3 id="__enter__"><code>__enter__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L133-L136">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__enter__()
</code></pre>




<h3 id="__exit__"><code>__exit__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/timed_threads.py#L138-L140">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__exit__(
    *args, **kwargs
)
</code></pre>






