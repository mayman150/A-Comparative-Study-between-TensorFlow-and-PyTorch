description: Abstract optimizer base class.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.Optimizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_variable"/>
<meta itemprop="property" content="add_variable_from_reference"/>
<meta itemprop="property" content="aggregate_gradients"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_gradients"/>
<meta itemprop="property" content="exclude_from_weight_decay"/>
<meta itemprop="property" content="finalize_variable_values"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="load_own_variables"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="save_own_variables"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="update_step"/>
</div>

# tf.keras.optimizers.Optimizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L876-L1365">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Abstract optimizer base class.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.optimizers.experimental.Optimizer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.Optimizer(
    name,
    weight_decay=0,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class supports distributed training. If you want to implement your own
optimizer, please subclass this class instead of _BaseOptimizer.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
String. The name to use
for momentum accumulator weights created by
the optimizer.
</td>
</tr><tr>
<td>
`weight_decay`<a id="weight_decay"></a>
</td>
<td>
Float, defaults to None. If set, weight decay is applied.
</td>
</tr><tr>
<td>
`clipnorm`<a id="clipnorm"></a>
</td>
<td>
Float. If set, the gradient of each weight is individually
clipped so that its norm is no higher than this value.
</td>
</tr><tr>
<td>
`clipvalue`<a id="clipvalue"></a>
</td>
<td>
Float. If set, the gradient of each weight is clipped to be no
higher than this value.
</td>
</tr><tr>
<td>
`global_clipnorm`<a id="global_clipnorm"></a>
</td>
<td>
Float. If set, the gradient of all weights is clipped so
that their global norm is no higher than this value.
</td>
</tr><tr>
<td>
`use_ema`<a id="use_ema"></a>
</td>
<td>
Boolean, defaults to False. If True, exponential moving average
(EMA) is applied. EMA consists of computing an exponential moving
average of the weights of the model (as the weight values change after
each training batch), and periodically overwriting the weights with
their moving average.
</td>
</tr><tr>
<td>
`ema_momentum`<a id="ema_momentum"></a>
</td>
<td>
Float, defaults to 0.99. Only used if `use_ema=True`.
This is the momentum to use when computing
the EMA of the model's weights:
`new_average = ema_momentum * old_average + (1 - ema_momentum) *
current_variable_value`.
</td>
</tr><tr>
<td>
`ema_overwrite_frequency`<a id="ema_overwrite_frequency"></a>
</td>
<td>
Int or None, defaults to None. Only used if
`use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
we overwrite the model variable by its moving average.
If None, the optimizer
does not overwrite model variables in the middle of training, and you
need to explicitly overwrite the variables at the end of training
by calling `optimizer.finalize_variable_values()`
(which updates the model
variables in-place). When using the built-in `fit()` training loop,
this happens automatically after the last epoch,
and you don't need to do anything.
</td>
</tr><tr>
<td>
`jit_compile`<a id="jit_compile"></a>
</td>
<td>
Boolean, defaults to True.
If True, the optimizer will use XLA
compilation. If no GPU device is found, this flag will be ignored.
</td>
</tr><tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>
optional <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a> instance. When provided,
the optimizer will be run in DTensor mode, e.g. state
tracking variable will be a DVariable, and aggregation/reduction will
happen in the global DTensor context.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
keyword arguments only used for backward compatibility.
</td>
</tr>
</table>


### Usage

```python
# Create an optimizer with the desired parameters.
opt = keras.optimizers.SGD(learning_rate=0.1)
var1, var2 = tf.Variable(1.0), tf.Variable(2.0)
# `loss` is a callable that takes no argument and returns the value
# to minimize.
loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
# Call minimize to update the list of variables.
opt.minimize(loss, var_list=[var1, var2])
```

### Processing gradients before applying them

Calling `minimize()` takes care of both computing the gradients and
applying them to the variables. If you want to process the gradients
before applying them you can instead use the optimizer in three steps:

1.  Compute the gradients with <a href="../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>.
2.  Process the gradients as you wish.
3.  Apply the processed gradients with `apply_gradients()`.

#### Example:



```python
# Create an optimizer.
opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
var1, var2 = tf.Variable(1.0), tf.Variable(2.0)

# Compute the gradients for a list of variables.
with tf.GradientTape() as tape:
  loss = 3 * var1 * var1 + 2 * var2 * var2
grads = tape.gradient(loss, [var1, var2])

# Process the gradients.
grads[0] = grads[0] + 1

# Ask the optimizer to apply the gradients on variables.
opt.apply_gradients(zip(grads, [var1, var2]))
```

### Dynamic learning rate

Dynamic learning rate can be achieved by setting learning rate as a built-in
or customized <a href="../../../tf/keras/optimizers/schedules/LearningRateSchedule.md"><code>tf.keras.optimizers.schedules.LearningRateSchedule</code></a>.

#### Example:



```
>>> var = tf.Variable(np.random.random(size=(1,)))
>>> learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
...   initial_learning_rate=.01, decay_steps=20, decay_rate=.1)
>>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate)
>>> loss = lambda: 3 * var
>>> opt.minimize(loss, var_list=[var])
```

### Gradients clipping

Users can clip the gradients before applying to variables by setting
`clipnorm`, `clipvalue` and `global_clipnorm`. Notice that `clipnorm` and
`global_clipnorm` can only have one being set.

#### Example:



```
>>> opt = tf.keras.optimizers.experimental.SGD(learning_rate=1, clipvalue=1)
>>> var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
>>> with tf.GradientTape() as tape:
...   loss = 2 * var1 + 2 * var2
>>> grads = tape.gradient(loss, [var1, var2])
>>> print([grads[0].numpy(), grads[1].numpy()])
[2.0, 2.0]
>>> opt.apply_gradients(zip(grads, [var1, var2]))
>>> # Without clipping, we should get [0, 0], but as gradients are clipped
>>> # to have max value 1, we get [1.0, 1.0].
>>> print([var1.numpy(), var2.numpy()])
[1.0, 1.0]
```

### Using weight decay.

Weight decay in certain scenarios can boost the model's performance. Keras
has built-in support for weight decay in all optimizers. Users can apply
weight decay by setting `weight_decay` argument.

```
>>> opt = tf.keras.optimizers.experimental.SGD(1, weight_decay=0.004)
>>> grads, var1, var2 = tf.zeros(()), tf.Variable(2.0), tf.Variable(2.0)
>>> # You can exclude variables from weight decay, in this case we
>>> # exclude `var2`.
>>> opt.exclude_from_weight_decay(var_list=[var2])
>>> opt.apply_gradients(zip([grads, grads], [var1, var2]))
>>> print([var1.numpy(), var2.numpy()])
[1.992, 2.0]
```


### Using exponential moving average.

Empirically it has been found that using the exponential moving average
(EMA) of the trained parameters of a deep network achieves a better
performance than using its trained parameters directly. Keras optimizers
allows users to compute this moving average and overwrite the model
variables at desired time.

#### Example:



```python
# Create an SGD optimizer with EMA on. `ema_momentum` controls the decay
# rate of the moving average. `ema_momentum=1` means no decay and the stored
# moving average is always model variable's initial value before training.
# Reversely, `ema_momentum=0` is equivalent to not using EMA.
# `ema_overwrite_frequency=3` means every 3 iterations, we overwrite the
# trainable variables with their moving average values.
opt = tf.keras.optimizers.experimental.SGD(
    learning_rate=1,
    use_ema=True,
    ema_momentum=0.5,
    ema_overwrite_frequency=3)
var1, var2 = tf.Variable(2.0), tf.Variable(2.0)
with tf.GradientTape() as tape:
  loss = var1 + var2
grads = tape.gradient(loss, [var1, var2])
# First iteration: [var1, var2] = [1.0, 1.0]
opt.apply_gradients(zip(grads, [var1, var2]))
print([var1, var2])

# Second iteration: [var1, var2] = [0.0, 0.0]
opt.apply_gradients(zip(grads, [var1, var2]))
print([var1, var2])

# Third iteration, without EMA, we should see [var1, var2] = [-1.0, -1.0],
# but overwriting results in [var1, var2] = [-0.125, -0.125]. The full
# calculation for the moving average of var1 is:
# var1=2*0.5**3+1*(1-0.5)*0.5**2+0*(1-0.5)*0.5**1+(-1)*(1-0.5)=-0.125.
opt.apply_gradients(zip(grads, [var1, var2]))
print([var1, var2])

```
When optimizer is constructed with `use_ema=True`, in custom training loop,
users can explicitly call `finalize_variable_values()` to overwrite
trainable variables with their EMA values. `finalize_variable_values()` is
by default called at the end of `model.fit()`.

### Use with <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>

This optimizer class is <a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> aware, which means it
automatically sums gradients across all replicas. To aggregate gradients
yourself, call `apply_gradients` with `skip_gradients_aggregation` set to
True.  This is useful if you need to process aggregated gradients.

```python
# This example is not runnable, it consists of dummy code for simple
# tutorial.
strategy = tf.distribute.experimental.TPUStrategy()

with strategy.scope():
  opt = tf.keras.optimizers.experimental.SGD()
  model = magic_function_that_returns_model()
  gradients = magic_function_that_returns_gradients()
  # Custom logic to aggregate gradients.
  gradients = strategy.reduce("SUM", gradients, axis=None)
  opt.apply_gradients(zip(gradients, model.trainable_variables),
      skip_gradients_aggregation=True)
```

### Creating a custom optimizer

If you intend to create your own optimization algorithm, please inherit from
this class and override the following methods:

  - `build`: Create your optimizer-related variables, such as `momentums` in
    SGD optimizer.
  - `update_step`: Implement your optimizer's updating logic.
  - `get_config`: serialization of the optimizer, include all hyper
    parameters.

Your optimizer would automatically be compatible with tensorflow distributed
training if you subclass `optimizer_experimental.Optimizer`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`iterations`<a id="iterations"></a>
</td>
<td>
The number of training steps this `optimizer` has run.

By default, iterations would be incremented by one every time
`apply_gradients()` is called.
</td>
</tr><tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`variables`<a id="variables"></a>
</td>
<td>
Returns variables of this optimizer.
</td>
</tr>
</table>



## Methods

<h3 id="add_variable"><code>add_variable</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L445-L470">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_variable(
    shape, dtype=None, initializer=&#x27;zeros&#x27;, name=None
)
</code></pre>

Create an optimizer variable.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`shape`
</td>
<td>
A list of integers, a tuple of integers, or a 1-D Tensor of
type int32. Defaults to scalar if unspecified.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The DType of the optimizer variable to be created. Defaults to
<a href="../../../tf/keras/backend/floatx.md"><code>tf.keras.backend.floatx</code></a> if unspecified.
</td>
</tr><tr>
<td>
`initializer`
</td>
<td>
string or callable. Initializer instance.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name of the optimizer variable to be created.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An optimizer variable, in the format of tf.Variable.
</td>
</tr>

</table>



<h3 id="add_variable_from_reference"><code>add_variable_from_reference</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L1099-L1127">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_variable_from_reference(
    model_variable, variable_name, shape=None, initial_value=None
)
</code></pre>

Create an optimizer variable from model variable.

Create an optimizer variable based on the information of model variable.
For example, in SGD optimizer momemtum, for each model variable, a
corresponding momemtum variable is created of the same shape and dtype.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_variable`
</td>
<td>
tf.Variable. The corresponding model variable to the
optimizer variable to be created.
</td>
</tr><tr>
<td>
`variable_name`
</td>
<td>
String. The name prefix of the optimizer variable to be
created. The create variables name will follow the pattern
`{variable_name}/{model_variable.name}`, e.g., `momemtum/dense_1`.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
List or Tuple, defaults to None. The shape of the optimizer
variable to be created. If None, the created variable will have the
same shape as `model_variable`.
</td>
</tr><tr>
<td>
`initial_value`
</td>
<td>
A Tensor, or Python object convertible to a Tensor,
defaults to None. The initial value of the optimizer variable, if
None, the initial value will be default to 0.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An optimizer variable.
</td>
</tr>

</table>



<h3 id="aggregate_gradients"><code>aggregate_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L1163-L1184">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>aggregate_gradients(
    grads_and_vars
)
</code></pre>

Aggregate gradients on all devices.

By default, we will perform reduce_sum of gradients across devices.
Users can implement their own aggregation logic by overriding this
method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`grads_and_vars`
</td>
<td>
List of (gradient, variable) pairs.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of (gradient, variable) pairs.
</td>
</tr>

</table>



<h3 id="apply_gradients"><code>apply_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L1186-L1223">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_gradients(
    grads_and_vars, name=None, skip_gradients_aggregation=False, **kwargs
)
</code></pre>

Apply gradients to variables.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`grads_and_vars`
</td>
<td>
List of `(gradient, variable)` pairs.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
string, defaults to None. The name of the namescope to
use when creating variables. If None, `self.name` will be used.
</td>
</tr><tr>
<td>
`skip_gradients_aggregation`
</td>
<td>
If true, gradients aggregation will not be
performed inside optimizer. Usually this arg is set to True when you
write custom code aggregating gradients outside the optimizer.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments only used for backward compatibility.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>, representing the current iteration.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If `grads_and_vars` is malformed.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If called in a cross-replica context.
</td>
</tr>
</table>



<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L402-L426">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>build(
    var_list
)
</code></pre>

Initialize the optimizer's variables, such as momemtum variables.

This function has to be implemented by subclass optimizers, and subclass
optimizers need to call `super().build(var_list)`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
List of model variables to build optimizers on. For example,
SGD optimizer with momentum will store one momentum variable
corresponding to each model variable.
</td>
</tr>
</table>



<h3 id="compute_gradients"><code>compute_gradients</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L243-L277">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_gradients(
    loss, var_list, tape=None
)
</code></pre>

Compute gradients of loss on trainable variables.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`loss`
</td>
<td>
`Tensor` or callable. If a callable, `loss` should take no
arguments and return the value to minimize.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
list or tuple of `Variable` objects to update to minimize
`loss`, or a callable returning the list or tuple of `Variable`
objects. Use callable when the variable list would otherwise be
incomplete before `minimize` since the variables are created at the
first time `loss` is called.
</td>
</tr><tr>
<td>
`tape`
</td>
<td>
(Optional) <a href="../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>. If `loss` is provided as a
`Tensor`, the tape that computed the `loss` must be provided.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of (gradient, variable) pairs. Variable is always present, but
gradient can be `None`.
</td>
</tr>

</table>



<h3 id="exclude_from_weight_decay"><code>exclude_from_weight_decay</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L567-L594">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>exclude_from_weight_decay(
    var_list=None, var_names=None
)
</code></pre>

Exclude variables from weight decay.

This method must be called before the optimizer's `build` method is
called. You can set specific variables to exclude out, or set a list of
strings as the anchor words, if any of which appear in a variable's
name, then the variable is excluded.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
A list of <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>s to exclude from weight decay.
</td>
</tr><tr>
<td>
`var_names`
</td>
<td>
A list of strings. If any string in `var_names` appear
in the model variable's name, then this model variable is
excluded from weight decay. For example, `var_names=['bias']`
excludes all bias variables from weight decay.
</td>
</tr>
</table>



<h3 id="finalize_variable_values"><code>finalize_variable_values</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L704-L717">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>finalize_variable_values(
    var_list
)
</code></pre>

Set the final value of model's trainable variables.

Sometimes there are some extra steps before ending the variable updates,
such as overriding the model variables with its average value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`var_list`
</td>
<td>
list of model variables.
</td>
</tr>
</table>



<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L759-L779">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config, custom_objects=None
)
</code></pre>

Creates an optimizer from its config.

This method is the reverse of `get_config`, capable of instantiating the
same optimizer from the config dictionary.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, typically the output of get_config.
</td>
</tr><tr>
<td>
`custom_objects`
</td>
<td>
A Python dictionary mapping names to additional
user-defined Python objects needed to recreate this optimizer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An optimizer instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L731-L757">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config of the optimizer.

An optimizer config is a Python dictionary (serializable)
containing the configuration of an optimizer.
The same optimizer can be reinstantiated later
(without any saved state) from this configuration.

Subclass optimizer should override this method to include other
hyperparameters.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary.
</td>
</tr>

</table>



<h3 id="load_own_variables"><code>load_own_variables</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L816-L832">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_own_variables(
    store
)
</code></pre>

Set the state of this optimizer object.


<h3 id="minimize"><code>minimize</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L522-L544">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>minimize(
    loss, var_list, tape=None
)
</code></pre>

Minimize `loss` by updating `var_list`.

This method simply computes gradient using <a href="../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> and calls
`apply_gradients()`. If you want to process the gradient before applying
then call <a href="../../../tf/GradientTape.md"><code>tf.GradientTape</code></a> and `apply_gradients()` explicitly instead
of using this function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`loss`
</td>
<td>
`Tensor` or callable. If a callable, `loss` should take no
arguments and return the value to minimize.
</td>
</tr><tr>
<td>
`var_list`
</td>
<td>
list or tuple of `Variable` objects to update to minimize
`loss`, or a callable returning the list or tuple of `Variable`
objects.  Use callable when the variable list would otherwise be
incomplete before `minimize` since the variables are created at the
first time `loss` is called.
</td>
</tr><tr>
<td>
`tape`
</td>
<td>
(Optional) <a href="../../../tf/GradientTape.md"><code>tf.GradientTape</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None
</td>
</tr>

</table>



<h3 id="save_own_variables"><code>save_own_variables</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L811-L814">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_own_variables(
    store
)
</code></pre>

Get the state of this optimizer object.


<h3 id="set_weights"><code>set_weights</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L786-L809">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>

Set the weights of the optimizer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`weights`
</td>
<td>
a list of <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>s or numpy arrays, the target values
of optimizer variables. It should have the same order as
`self._variables`.
</td>
</tr>
</table>



<h3 id="update_step"><code>update_step</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/optimizer.py#L191-L205">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>update_step(
    gradient, variable
)
</code></pre>

Function to update variable value based on given gradients.

This method must be implemented in customized optimizers.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`gradient`
</td>
<td>
backpropagated gradient of the given variable.
</td>
</tr><tr>
<td>
`variable`
</td>
<td>
variable whose value needs to be updated.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `Operation` that applies the specified gradients.
</td>
</tr>

</table>





