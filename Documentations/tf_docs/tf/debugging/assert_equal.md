description: Assert the condition x == y holds element-wise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.assert_equal" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.assert_equal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/check_ops.py">View source</a>



Assert the condition `x == y` holds element-wise.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.assert_equal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.assert_equal(
    x, y, message=None, summarize=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This Op checks that `x[i] == y[i]` holds for every pair of (possibly
broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is
trivially satisfied.

If `x` == `y` does not hold, `message`, as well as the first `summarize`
entries of `x` and `y` are printed, and `InvalidArgumentError` is raised.

When using inside <a href="../../tf/function.md"><code>tf.function</code></a>, this API takes effects during execution.
It's recommended to use this API with <a href="../../tf/control_dependencies.md"><code>tf.control_dependencies</code></a> to
ensure the correct execution order.

In the following example, without <a href="../../tf/control_dependencies.md"><code>tf.control_dependencies</code></a>, errors may
not be raised at all.
Check <a href="../../tf/control_dependencies.md"><code>tf.control_dependencies</code></a> for more details.

```
>>> def check_size(x):
...   with tf.control_dependencies([
...       tf.debugging.assert_equal(tf.size(x), 3,
...                       message='Bad tensor size')]):
...     return x
```

```
>>> check_size(tf.ones([2, 3], tf.float32))
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
 Numeric `Tensor`.
</td>
</tr><tr>
<td>
`y`<a id="y"></a>
</td>
<td>
 Numeric `Tensor`, same dtype as and broadcastable to `x`.
</td>
</tr><tr>
<td>
`message`<a id="message"></a>
</td>
<td>
A string to prefix to the default message. (optional)
</td>
</tr><tr>
<td>
`summarize`<a id="summarize"></a>
</td>
<td>
Print this many entries of each tensor. (optional)
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this operation (optional).  Defaults to "assert_equal".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Op that raises `InvalidArgumentError` if `x == y` is False. This can
be used with <a href="../../tf/control_dependencies.md"><code>tf.control_dependencies</code></a> inside of <a href="../../tf/function.md"><code>tf.function</code></a>s to
block followup computation until the check has executed.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`<a id="InvalidArgumentError"></a>
</td>
<td>
if the check can be performed immediately and
`x == y` is False. The check can be performed immediately during eager
execution or if `x` and `y` are statically known.
</td>
</tr>
</table>



 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

returns None

 </devsite-expandable></section>

