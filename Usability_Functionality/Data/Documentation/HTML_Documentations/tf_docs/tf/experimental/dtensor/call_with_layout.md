description: Calls a function in the DTensor device scope if layout is not None.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.call_with_layout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.call_with_layout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Calls a function in the DTensor device scope if `layout` is not None.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.call_with_layout(
    fn: Callable[..., Any],
    layout: Optional[<a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>],
    *args,
    **kwargs
) -> Any
</code></pre>



<!-- Placeholder for "Used in" -->

If `layout` is not None, `fn` consumes DTensor(s) as input and produces a
DTensor as output; a DTensor is a tf.Tensor with layout-related attributes.

If `layout` is None, `fn` consumes and produces regular tf.Tensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fn`<a id="fn"></a>
</td>
<td>
A supported TF API function such as tf.zeros.
</td>
</tr><tr>
<td>
`layout`<a id="layout"></a>
</td>
<td>
Optional, the layout of the output DTensor.
</td>
</tr><tr>
<td>
`*args`<a id="*args"></a>
</td>
<td>
 Arguments given to `fn`.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Keyword arguments given to `fn`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The return value of `fn` transformed to a DTensor if requested.
</td>
</tr>

</table>

