description: A LearningRateSchedule that uses a cosine decay with optional warmup.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.schedules.CosineDecay" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.optimizers.schedules.CosineDecay

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/schedules/learning_rate_schedule.py#L578-L767">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A LearningRateSchedule that uses a cosine decay with optional warmup.

Inherits From: [`LearningRateSchedule`](../../../../tf/keras/optimizers/schedules/LearningRateSchedule.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.experimental.CosineDecay`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps,
    alpha=0.0,
    name=None,
    warmup_target=None,
    warmup_steps=0
)
</code></pre>



<!-- Placeholder for "Used in" -->

See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
SGDR: Stochastic Gradient Descent with Warm Restarts.

For the idea of a linear warmup of our learning rate,
see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

When we begin training a model, we often want an initial increase in our
learning rate followed by a decay. If `warmup_target` is an int, this
schedule applies a linear increase per optimizer step to our learning rate
from `initial_learning_rate` to `warmup_target` for a duration of
`warmup_steps`. Afterwards, it applies a cosine decay function taking our
learning rate from `warmup_target` to `alpha` for a duration of
`decay_steps`. If `warmup_target` is None we skip warmup and our decay
will take our learning rate from `initial_learning_rate` to `alpha`.
It requires a `step` value to  compute the learning rate. You can
just pass a TensorFlow variable that you increment at each training step.

The schedule is a 1-arg callable that produces a warmup followed by a
decayed learning rate when passed the current optimizer step. This can be
useful for changing the learning rate value across different invocations of
optimizer functions.

Our warmup is computed as:

```python
def warmup_learning_rate(step):
    completed_fraction = step / warmup_steps
    total_delta = target_warmup - initial_learning_rate
    return completed_fraction * total_delta
```

And our decay is computed as:

```python
if warmup_target is None:
    initial_decay_lr = initial_learning_rate
else:
    initial_decay_lr = warmup_target

def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_decay_lr * decayed
```

Example usage without warmup:

```python
decay_steps = 1000
initial_learning_rate = 0.1
lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps)
```

Example usage with warmup:

```python
decay_steps = 1000
initial_learning_rate = 0
warmup_steps = 1000
target_learning_rate = 0.1
lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
    warmup_steps=warmup_steps
)
```

You can pass this schedule directly into a <a href="../../../../tf/keras/optimizers/Optimizer.md"><code>tf.keras.optimizers.Optimizer</code></a>
as the learning rate. The learning rate schedule is also serializable and
deserializable using <a href="../../../../tf/keras/optimizers/schedules/serialize.md"><code>tf.keras.optimizers.schedules.serialize</code></a> and
<a href="../../../../tf/keras/optimizers/schedules/deserialize.md"><code>tf.keras.optimizers.schedules.deserialize</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar `Tensor` of the same
type as `initial_learning_rate`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`initial_learning_rate`<a id="initial_learning_rate"></a>
</td>
<td>
A scalar `float32` or `float64` `Tensor` or a
Python int. The initial learning rate.
</td>
</tr><tr>
<td>
`decay_steps`<a id="decay_steps"></a>
</td>
<td>
A scalar `int32` or `int64` `Tensor` or a Python int.
Number of steps to decay over.
</td>
</tr><tr>
<td>
`alpha`<a id="alpha"></a>
</td>
<td>
A scalar `float32` or `float64` `Tensor` or a Python int.
Minimum learning rate value for decay as a fraction of
`initial_learning_rate`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
String. Optional name of the operation.  Defaults to
'CosineDecay'.
</td>
</tr><tr>
<td>
`warmup_target`<a id="warmup_target"></a>
</td>
<td>
None or a scalar `float32` or `float64` `Tensor` or a
Python int. The target learning rate for our warmup phase. Will cast
to the `initial_learning_rate` datatype. Setting to None will skip
warmup and begins decay phase from `initial_learning_rate`.
Otherwise scheduler will warmup from `initial_learning_rate` to
`warmup_target`.
</td>
</tr><tr>
<td>
`warmup_steps`<a id="warmup_steps"></a>
</td>
<td>
A scalar `int32` or `int64` `Tensor` or a Python int.
Number of steps to warmup over.
</td>
</tr>
</table>



## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/schedules/learning_rate_schedule.py#L88-L98">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates a `LearningRateSchedule` from its config.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
Output of `get_config()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `LearningRateSchedule` instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/schedules/learning_rate_schedule.py#L759-L767">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/schedules/learning_rate_schedule.py#L718-L757">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    step
)
</code></pre>

Call self as a function.




