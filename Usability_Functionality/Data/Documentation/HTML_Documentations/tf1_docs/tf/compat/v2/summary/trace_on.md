<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v2.summary.trace_on" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v2.summary.trace_on

Starts a trace to record computation graphs and profiling information.

``` python
tf.compat.v2.summary.trace_on(
    graph=True,
    profiler=False
)
```

<!-- Placeholder for "Used in" -->

Must be invoked in eager mode.

When enabled, TensorFlow runtime will collection information that can later be
exported and consumed by TensorBoard. The trace is activated across the entire
TensorFlow runtime and affects all threads of execution.

To stop the trace and export the collected information, use
`tf.summary.trace_export`. To stop the trace without exporting, use
`tf.summary.trace_off`.

#### Args:


* <b>`graph`</b>: If True, enables collection of executed graphs. It includes ones from
    tf.function invocation and ones from the legacy graph mode. The default
    is True.
* <b>`profiler`</b>: If True, enables the advanced profiler. Enabling profiler
    implicitly enables the graph collection. The profiler may incur a high
    memory overhead. The default is False.