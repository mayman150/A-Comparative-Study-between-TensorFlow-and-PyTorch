description: Gets an existing variable with these parameters or create a new one.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.get_variable" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.get_variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/variable_scope.py">View source</a>



Gets an existing variable with these parameters or create a new one.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=<a href="../../../tf/VariableSynchronization.md#AUTO"><code>tf.VariableSynchronization.AUTO</code></a>,
    aggregation=<a href="../../../tf/compat/v1/VariableAggregation.md#NONE"><code>tf.compat.v1.VariableAggregation.NONE</code></a>
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

Although it is a legacy <a href="../../../tf/compat/v1.md"><code>compat.v1</code></a> api,
<a href="../../../tf/compat/v1/get_variable.md"><code>tf.compat.v1.get_variable</code></a> is mostly compatible with eager
execution and <a href="../../../tf/function.md"><code>tf.function</code></a> but only if you combine it with the
`tf.compat.v1.keras.utils.track_tf1_style_variables` decorator. (Though
it will behave as if reuse is always set to `AUTO_REUSE`.)

See the
[model migration guide](https://www.tensorflow.org/guide/migrate/model_mapping)
for more info.

If you do not combine it with
`tf.compat.v1.keras.utils.track_tf1_style_variables`, `get_variable` will create
a brand new variable every single time it is called and will never reuse
variables, regardless of variable names or `reuse` arguments.

The TF2 equivalent of this symbol would be <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>, but note
that when using <a href="../../../tf/Variable.md"><code>tf.Variable</code></a> you must make sure you track your variables
(and regularizer arguments) either manually or via <a href="../../../tf/Module.md"><code>tf.Module</code></a> or
<a href="../../../tf/keras/layers/Layer.md"><code>tf.keras.layers.Layer</code></a> mechanisms.

A section of the
[migration guide](https://www.tensorflow.org/guide/migrate/model_mapping#incremental_migration_to_native_tf2)
provides more details on incrementally migrating these usages to <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>
as well.

Note: The `partitioner` arg is not compatible with TF2 behaviors even when
using `tf.compat.v1.keras.utils.track_tf1_style_variables`. It can be replaced
by using `ParameterServerStrategy` and its partitioners. See the
[multi-gpu migration guide](https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training)
and the ParameterServerStrategy guides it references for more info.

 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->



This function prefixes the name with the current variable scope
and performs reuse checks. See the
[Variable Scope How To](https://tensorflow.org/guide/variables)
for an extensive description of how reusing works. Here is a basic example:

```python
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```

If initializer is `None` (the default), the default initializer passed in
the variable scope will be used. If that one is `None` too, a
`glorot_uniform_initializer` will be used. The initializer can also be
a Tensor, in which case the variable is initialized to this value and shape.

Similarly, if the regularizer is `None` (the default), the default regularizer
passed in the variable scope will be used (if that is `None` too,
then by default no regularization is performed).

If a partitioner is provided, a `PartitionedVariable` is returned.
Accessing this object as a `Tensor` returns the shards concatenated along
the partition axis.

Some useful partitioners are available.  See, e.g.,
`variable_axis_size_partitioner` and `min_max_variable_partitioner`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name of the new or existing variable.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
Shape of the new or existing variable.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Type of the new or existing variable (defaults to `DT_FLOAT`).
</td>
</tr><tr>
<td>
`initializer`<a id="initializer"></a>
</td>
<td>
Initializer for the variable if one is created. Can either be
an initializer object or a Tensor. If it's a Tensor, its shape must be known
unless validate_shape is False.
</td>
</tr><tr>
<td>
`regularizer`<a id="regularizer"></a>
</td>
<td>
A (Tensor -> Tensor or None) function; the result of
applying it on a newly created variable will be added to the collection
`tf.GraphKeys.REGULARIZATION_LOSSES` and can be used for regularization.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
If `True` also add the variable to the graph collection
`GraphKeys.TRAINABLE_VARIABLES` (see <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>).
</td>
</tr><tr>
<td>
`collections`<a id="collections"></a>
</td>
<td>
List of graph collections keys to add the Variable to.
Defaults to `[GraphKeys.GLOBAL_VARIABLES]` (see <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>).
</td>
</tr><tr>
<td>
`caching_device`<a id="caching_device"></a>
</td>
<td>
Optional device string or function describing where the
Variable should be cached for reading.  Defaults to the Variable's
device.  If not `None`, caches on another device.  Typical use is to
cache on the device where the Ops using the Variable reside, to
deduplicate copying through `Switch` and other conditional statements.
</td>
</tr><tr>
<td>
`partitioner`<a id="partitioner"></a>
</td>
<td>
Optional callable that accepts a fully defined `TensorShape`
and `dtype` of the Variable to be created, and returns a list of
partitions for each axis (currently only one axis can be partitioned).
</td>
</tr><tr>
<td>
`validate_shape`<a id="validate_shape"></a>
</td>
<td>
If False, allows the variable to be initialized with a
value of unknown shape. If True, the default, the shape of initial_value
must be known. For this to be used the initializer must be a Tensor and
not an initializer object.
</td>
</tr><tr>
<td>
`use_resource`<a id="use_resource"></a>
</td>
<td>
If False, creates a regular Variable. If true, creates an
experimental ResourceVariable instead with well-defined semantics.
Defaults to False (will later change to True). When eager execution is
enabled this argument is always forced to be True.
</td>
</tr><tr>
<td>
`custom_getter`<a id="custom_getter"></a>
</td>
<td>
Callable that takes as a first argument the true getter, and
allows overwriting the internal get_variable method.
The signature of `custom_getter` should match that of this method,
but the most future-proof version will allow for changes:
`def custom_getter(getter, *args, **kwargs)`.  Direct access to
all `get_variable` parameters is also allowed:
`def custom_getter(getter, name, *args, **kwargs)`.  A simple identity
custom getter that simply creates variables with modified names is:
```python
def custom_getter(getter, name, *args, **kwargs):
  return getter(name + '_suffix', *args, **kwargs)
```
</td>
</tr><tr>
<td>
`constraint`<a id="constraint"></a>
</td>
<td>
An optional projection function to be applied to the variable
after being updated by an `Optimizer` (e.g. used to implement norm
constraints or value constraints for layer weights). The function must
take as input the unprojected Tensor representing the value of the
variable and return the Tensor for the projected value
(which must have the same shape). Constraints are not safe to
use when doing asynchronous distributed training.
</td>
</tr><tr>
<td>
`synchronization`<a id="synchronization"></a>
</td>
<td>
Indicates when a distributed a variable will be
aggregated. Accepted values are constants defined in the class
<a href="../../../tf/VariableSynchronization.md"><code>tf.VariableSynchronization</code></a>. By default the synchronization is set to
`AUTO` and the current `DistributionStrategy` chooses
when to synchronize.
</td>
</tr><tr>
<td>
`aggregation`<a id="aggregation"></a>
</td>
<td>
Indicates how a distributed variable will be aggregated.
Accepted values are constants defined in the class
<a href="../../../tf/VariableAggregation.md"><code>tf.VariableAggregation</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The created or existing `Variable` (or `PartitionedVariable`, if a
partitioner was used).
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
when creating a new variable and shape is not declared,
when violating reuse during variable creation, or when `initializer` dtype
and `dtype` don't match. Reuse is set inside `variable_scope`.
</td>
</tr>
</table>

