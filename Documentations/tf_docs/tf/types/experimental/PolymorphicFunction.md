description: Base class for polymorphic graph functions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.PolymorphicFunction" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="experimental_get_compiler_ir"/>
<meta itemprop="property" content="get_concrete_function"/>
</div>

# tf.types.experimental.PolymorphicFunction

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>



Base class for polymorphic graph functions.

Inherits From: [`Callable`](../../../tf/types/experimental/Callable.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.types.experimental.GenericFunction`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Graph functions are Python callable objects that dispatch calls to a
TensorFlow graph. Polymorphic graph functions can be backed by multiple TF
graphs, and automatically select the appropriate specialization based on the
type of input they were called with. They may also create specializations on
the fly if necessary, for example by tracing.

Also see <a href="../../../tf/function.md"><code>tf.function</code></a>.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`function_type`<a id="function_type"></a>
</td>
<td>
Returns a FunctionType describing this callable.
</td>
</tr>
</table>



## Methods

<h3 id="experimental_get_compiler_ir"><code>experimental_get_compiler_ir</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>experimental_get_compiler_ir(
    *args, **kwargs
)
</code></pre>

Returns compiler IR for the compiled function.

This API is intended *only* for debugging as there are no guarantees on
backwards compatibility of returned IR or the allowed values of `stage`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
compilation args supports inputs either: (1) all inputs are
TensorSpec or (2) all inputs are tf.Tensor/Python variables.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments used for compilation. Same requirement as
compiliation args.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Function callable with the following kwargs:
  - `stage` at which the compiler IR should be serialized. Allowed values
    are:
     - `hlo`: HLO output after conversion from TF
      (https://www.tensorflow.org/xla/operation_semantics).
     - `hlo_serialized`: Like stage=`hlo`, but the output is a serialized
       HLO module proto (a bytes object).
     - `optimized_hlo`: HLO after compiler optimizations.
     - `optimized_hlo_serialized`: Like stage=`optimized_hlo`, but the
       output is a serialized HLO module proto (a bytes object).
     - `optimized_hlo_dot`: optimized HLO in DOT format suitable for
       Graphviz.
  - `device_name` can be either None, in which case the preferred device
    is used for compilation, or a device name. It can be a full device
    name, or a partial one, e.g., `/device:CPU:0`.

For example, for

```python
@tf.function(jit_compile=True)
def f(x):
  return x + 1

f.experimental_get_compiler_ir(tf.random.normal([10, 10])(stage='hlo')
```

the output is:

```
HloModule a_inference_f_13__.9

ENTRY %a_inference_f_13__.9 (arg0.1: f32[10,10]) -> f32[10,10] {
  %arg0.1 = f32[10,10]{1,0} parameter(0), parameter_replication={false}
  %reshape.2 = f32[10,10]{1,0} reshape(f32[10,10]{1,0} %arg0.1)
  %constant.3 = f32[] constant(1)
  %broadcast.4 = f32[10,10]{1,0} broadcast(f32[] %constant.3)
  %add.5 = f32[10,10]{1,0} add(f32[10,10]{1,0} %reshape.2,
                               f32[10,10]{1,0} %broadcast.4)
  %reshape.6 = f32[10,10]{1,0} reshape(f32[10,10]{1,0} %add.5)
  %tuple.7 = (f32[10,10]{1,0}) tuple(f32[10,10]{1,0} %reshape.6)
  ROOT %get-tuple-element.8 = f32[10,10]{1,0}
    get-tuple-element((f32[10,10]{1,0}) %tuple.7), index=0
}
```

Here is another example using tf.TensorSpec inputs:

```python
y = tf.Variable(tf.zeros([10, 20], dtype=tf.float32))

@tf.function(jit_compile=True)
def f(x):
  return x + y

hlo_str = f.experimental_get_compiler_ir(tf.TensorSpec(shape=(10,
20)))(stage='hlo')
```

The output is:

```
HloModule a_inference_f_120__.8,
entry_computation_layout={(f32[10,20]{1,0},f32[10,20]{1,0})->f32[10,20]{1,0}}

ENTRY %a_inference_f_120__.8 (arg0.1: f32[10,20], arg1.2: f32[10,20]) ->
f32[10,20] {
  %arg0.1 = f32[10,20]{1,0} parameter(0), parameter_replication={false},
  metadata={op_name="XLA_Args"}
  %reshape.3 = f32[10,20]{1,0} reshape(f32[10,20]{1,0} %arg0.1)
  %arg1.2 = f32[10,20]{1,0} parameter(1), parameter_replication={false},
  metadata={op_name="XLA_Args"}
  %add.4 = f32[10,20]{1,0} add(f32[10,20]{1,0} %reshape.3, f32[10,20]{1,0}
  %arg1.2), metadata={op_type="AddV2" op_name="add"
  source_file="<ipython-input-16-ea04879c1873>" source_line=4}
  %reshape.5 = f32[10,20]{1,0} reshape(f32[10,20]{1,0} %add.4),
  metadata={op_name="XLA_Retvals"}
  %tuple.6 = (f32[10,20]{1,0}) tuple(f32[10,20]{1,0} %reshape.5),
  metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.7 = f32[10,20]{1,0}
  get-tuple-element((f32[10,20]{1,0}) %tuple.6), index=0,
  metadata={op_name="XLA_Retvals"}
}
</td>
</tr>

</table>


```

The HLO module accepts a flat list of inputs. To retrieve the order
of these inputs signatures, users can call the
`concrete_fn.structured_input_signature` and `concrete_fn.captured_inputs`:

```python
# Use concrete_fn to get the hlo_module flat_args.
concrete_fn = f.get_concrete_function(tf.TensorSpec(shape=(10, 20)))
flat_args = list(
    tf.nest.flatten(concrete_fn.structured_input_signature)
    ) + concrete_fn.captured_inputs
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
  (1) If an invalid `stage` is selected
(2) or if applied to a function which is not compiled
(`jit_compile=True` is not set).
(3) or if input shapes are not fully defined for tf.TensorSpec inputs
</td>
</tr><tr>
<td>
`TypeError`
</td>
<td>
When called with input in graph mode.
</td>
</tr>
</table>



<h3 id="get_concrete_function"><code>get_concrete_function</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get_concrete_function(
    *args, **kwargs
) -> <a href="../../../tf/types/experimental/ConcreteFunction.md"><code>tf.types.experimental.ConcreteFunction</code></a>
</code></pre>

Returns a `ConcreteFunction` specialized to input types.

The arguments specified by `args` and `kwargs` follow normal function call
rules. The returned `ConcreteFunction` has the same set of positional and
keyword arguments as `self`, but their types are compatible to the types
specified by `args` and `kwargs` (though not neccessarily equal).

```
>>> @tf.function
... def f(x):
...   return x
>>> f_concrete = f.get_concrete_function(tf.constant(1.0))
>>> f_concrete = f.get_concrete_function(x=tf.constant(1.0))
```

Unlike normal calls, `get_concrete_function` allow type specifiers instead
of TensorFlow objects, so for example <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>s may be replaced with
<a href="../../../tf/TensorSpec.md"><code>tf.TensorSpec</code></a>s.

```
>>> @tf.function
... def f(x):
...   return x
>>> f_concrete = f.get_concrete_function(tf.TensorSpec([], tf.float64))
```

If the function definition allows only one specialization, `args` and
`kwargs` may be omitted altogether.

```
>>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
... def f(x):
...   return x
>>> f_concrete = f.get_concrete_function()
```

The returned `ConcreteFunction` can be called normally:

```
>>> f_concrete(tf.constant(1.0))
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
>>> f_concrete(x=tf.constant(1.0))
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
inputs to specialize on.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
inputs to specialize on.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `ConcreteFunction`.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Executes this callable.

This behaves like a regular op - in eager mode, it immediately starts
execution, returning results. In graph mode, it creates ops which return
symbolic TensorFlow values (like <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>,
etc.). For example, <a href="../../../tf/function.md"><code>tf.function</code></a> callables typically generate a
<a href="../../../tf/raw_ops/PartitionedCall.md"><code>tf.raw_ops.PartitionedCall</code></a> op, but not always - the
exact operations being generated are an internal implementation detail.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*args`
</td>
<td>
positional argument for this call
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments for this call
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The execution results.
</td>
</tr>

</table>





