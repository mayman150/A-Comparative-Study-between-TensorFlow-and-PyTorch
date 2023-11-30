description: A protocol allowing custom classes to control tf.function retracing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.SupportsTracingProtocol" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__tf_tracing_type__"/>
</div>

# tf.types.experimental.SupportsTracingProtocol

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>



A protocol allowing custom classes to control tf.function retracing.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.types.experimental.SupportsTracingProtocol(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="__tf_tracing_type__"><code>__tf_tracing_type__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__tf_tracing_type__(
    context: TracingContext
) -> <a href="../../../tf/types/experimental/TraceType.md"><code>tf.types.experimental.TraceType</code></a>
</code></pre>

Returns the tracing type of this object.

The tracing type is used to build the signature of a tf.function
when traced, and to match arguments with existing signatures.
When a Function object is called, tf.function looks at the tracing type
of the call arguments. If an existing signature of matching type exists,
it will be used. Otherwise, a new function is traced, and its signature
will use the tracing type of the call arguments.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`context`
</td>
<td>
a context reserved for internal/future usage.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The tracing type of this object.
</td>
</tr>

</table>





