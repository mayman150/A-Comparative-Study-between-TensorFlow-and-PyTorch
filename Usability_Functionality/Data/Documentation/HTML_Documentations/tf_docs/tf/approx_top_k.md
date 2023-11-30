description: Returns min/max k values and their indices of the input operand in an approximate manner.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.approx_top_k" />
<meta itemprop="path" content="Stable" />
</div>

# tf.approx_top_k

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns min/max k values and their indices of the input operand in an approximate manner.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.approx_top_k`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.approx_top_k(
    input: Annotated[Any, TV_ApproxTopK_T],
    k: int,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    is_max_k: bool = True,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: bool = True,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

See https://arxiv.org/abs/2206.14286 for the algorithm details.
This op is only optimized on TPU currently.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
Array to search. Must be at least 1-D of the floating type
</td>
</tr><tr>
<td>
`k`<a id="k"></a>
</td>
<td>
An `int` that is `>= 0`. Specifies the number of min/max-k.
</td>
</tr><tr>
<td>
`reduction_dimension`<a id="reduction_dimension"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
Integer dimension along which to search. Default: -1.
</td>
</tr><tr>
<td>
`recall_target`<a id="recall_target"></a>
</td>
<td>
An optional `float`. Defaults to `0.95`.
Recall target for the approximation. Range in (0,1]
</td>
</tr><tr>
<td>
`is_max_k`<a id="is_max_k"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
When true, computes max-k; otherwise computes min-k.
</td>
</tr><tr>
<td>
`reduction_input_size_override`<a id="reduction_input_size_override"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
When set to a positive value, it overrides the size determined by
`input[reduction_dim]` for evaluating the recall. This option is useful when
the given `input` is only a subset of the overall computation in SPMD or
distributed pipelines, where the true input size cannot be deferred by the
`input` shape.
</td>
</tr><tr>
<td>
`aggregate_to_topk`<a id="aggregate_to_topk"></a>
</td>
<td>
An optional `bool`. Defaults to `True`.
When true, aggregates approximate results to top-k. When false, returns the
approximate results. The number of the approximate results is implementation
defined and is greater equals to the specified `k`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of `Tensor` objects (values, indices).
</td>
</tr>
<tr>
<td>
`values`<a id="values"></a>
</td>
<td>
A `Tensor`. Has the same type as `input`.
</td>
</tr><tr>
<td>
`indices`<a id="indices"></a>
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr>
</table>

