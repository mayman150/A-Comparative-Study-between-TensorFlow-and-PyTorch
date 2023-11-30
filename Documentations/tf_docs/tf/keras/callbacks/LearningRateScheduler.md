description: Learning rate scheduler.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.LearningRateScheduler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.LearningRateScheduler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L2227-L2296">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Learning rate scheduler.

Inherits From: [`Callback`](../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.LearningRateScheduler(
    schedule, verbose=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

At the beginning of every epoch, this callback gets the updated learning
rate value from `schedule` function provided at `__init__`, with the current
epoch and current learning rate, and applies the updated learning rate on
the optimizer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`schedule`<a id="schedule"></a>
</td>
<td>
a function that takes an epoch index (integer, indexed from 0)
and current learning rate (float) as inputs and returns a new
learning rate as output (float).
</td>
</tr><tr>
<td>
`verbose`<a id="verbose"></a>
</td>
<td>
int. 0: quiet, 1: update messages.
</td>
</tr>
</table>



#### Example:



```
>>> # This function keeps the initial learning rate for the first ten epochs
>>> # and decreases it exponentially after that.
>>> def scheduler(epoch, lr):
...   if epoch < 10:
...     return lr
...   else:
...     return lr * tf.math.exp(-0.1)
>>>
>>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
>>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
>>> round(model.optimizer.lr.numpy(), 5)
0.01
```

```
>>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
>>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
...                     epochs=15, callbacks=[callback], verbose=0)
>>> round(model.optimizer.lr.numpy(), 5)
0.00607
```

## Methods

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






