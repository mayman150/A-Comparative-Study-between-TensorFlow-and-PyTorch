description: Base class for graph functions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.AtomicFunction" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="call_with_captures"/>
</div>

# tf.types.experimental.AtomicFunction

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>



Base class for graph functions.

Inherits From: [`Callable`](../../../tf/types/experimental/Callable.md)

<!-- Placeholder for "Used in" -->

An `AtomicFunction` encapsulates a single graph function definition.

`AtomicFunction` can be called directly only if no captures are needed
according to the `FunctionType`. If captures are present, please use
`call_with_captures` instead.

`AtomicFunction` does not support gradients. Please use the parent
`ConcreteFunction` if you need gradient support.



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

<h3 id="call_with_captures"><code>call_with_captures</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/core.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call_with_captures(
    args, kwargs, captures
)
</code></pre>

Calls this AtomicFunction with captures as defined by its FunctionType.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`args`
</td>
<td>
Tuple containing positional arguments
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Dict containing keyword arguments
</td>
</tr><tr>
<td>
`captures`
</td>
<td>
Tuple of tensors supplying captured tensor values.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A structured output value based on the inputs.
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





