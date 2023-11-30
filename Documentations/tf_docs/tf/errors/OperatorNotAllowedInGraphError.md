description: Raised when an unsupported operator is present in Graph execution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.errors.OperatorNotAllowedInGraphError" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.errors.OperatorNotAllowedInGraphError

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/errors_impl.py">View source</a>



Raised when an unsupported operator is present in Graph execution.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.errors.OperatorNotAllowedInGraphError(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

For example, using a <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> as a Python `bool` inside a Graph will
raise `OperatorNotAllowedInGraphError`. Iterating over values inside a
<a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> is also not supported in Graph execution.

#### Example:


```
>>> @tf.function
... def iterate_over(t):
...   a,b,c = t
...   return a
>>>
>>> iterate_over(tf.constant([1, 2, 3]))
Traceback (most recent call last):
...
OperatorNotAllowedInGraphError: ...
```

