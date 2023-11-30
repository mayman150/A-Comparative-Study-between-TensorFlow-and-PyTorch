description: Computes the LSTM cell forward propagation for 1 time step.
robots: noindex

# tf.raw_ops.LSTMBlockCell

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the LSTM cell forward propagation for 1 time step.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.LSTMBlockCell`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.LSTMBlockCell(
    x,
    cs_prev,
    h_prev,
    w,
    wci,
    wcf,
    wco,
    b,
    forget_bias=1,
    cell_clip=3,
    use_peephole=False,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This implementation uses 1 weight matrix and 1 bias vector, and there's an
optional peephole connection.

This kernel op implements the following mathematical equations:

```python
xh = [x, h_prev]
[i, f, ci, o] = xh * w + b
f = f + forget_bias

if not use_peephole:
  wci = wcf = wco = 0

i = sigmoid(cs_prev * wci + i)
f = sigmoid(cs_prev * wcf + f)
ci = tanh(ci)

cs = ci .* i + cs_prev .* f
cs = clip(cs, cell_clip)

o = sigmoid(cs * wco + o)
co = tanh(cs)
h = co .* o
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
A `Tensor`. Must be one of the following types: `half`, `float32`.
The input to the LSTM cell, shape (batch_size, num_inputs).
</td>
</tr><tr>
<td>
`cs_prev`<a id="cs_prev"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
Value of the cell state at previous time step.
</td>
</tr><tr>
<td>
`h_prev`<a id="h_prev"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
Output of the previous cell at previous time step.
</td>
</tr><tr>
<td>
`w`<a id="w"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The weight matrix.
</td>
</tr><tr>
<td>
`wci`<a id="wci"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
The weight matrix for input gate peephole connection.
</td>
</tr><tr>
<td>
`wcf`<a id="wcf"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
The weight matrix for forget gate peephole connection.
</td>
</tr><tr>
<td>
`wco`<a id="wco"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
The weight matrix for output gate peephole connection.
</td>
</tr><tr>
<td>
`b`<a id="b"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The bias vector.
</td>
</tr><tr>
<td>
`forget_bias`<a id="forget_bias"></a>
</td>
<td>
An optional `float`. Defaults to `1`. The forget gate bias.
</td>
</tr><tr>
<td>
`cell_clip`<a id="cell_clip"></a>
</td>
<td>
An optional `float`. Defaults to `3`.
Value to clip the 'cs' value to.
</td>
</tr><tr>
<td>
`use_peephole`<a id="use_peephole"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
Whether to use peephole weights.
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
A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).
</td>
</tr>
<tr>
<td>
`i`<a id="i"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`cs`<a id="cs"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`f`<a id="f"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`o`<a id="o"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`ci`<a id="ci"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`co`<a id="co"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`h`<a id="h"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr>
</table>

