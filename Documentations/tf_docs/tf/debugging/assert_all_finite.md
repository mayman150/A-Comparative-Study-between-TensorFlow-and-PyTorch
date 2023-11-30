description: Assert that the tensor does not contain any NaN's or Inf's.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.assert_all_finite" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.assert_all_finite

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/numerics.py">View source</a>



Assert that the tensor does not contain any NaN's or Inf's.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.assert_all_finite(
    x, message, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

```
>>> @tf.function
... def f(x):
...   x = tf.debugging.assert_all_finite(x, 'Input x must be all finite')
...   return x + 1
```

```
>>> f(tf.constant([np.inf, 1, 2]))
Traceback (most recent call last):
   ...
InvalidArgumentError: ...
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
Tensor to check.
</td>
</tr><tr>
<td>
`message`<a id="message"></a>
</td>
<td>
Message to log on failure.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Same tensor as `x`.
</td>
</tr>

</table>

