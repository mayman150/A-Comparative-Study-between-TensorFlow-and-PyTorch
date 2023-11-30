description: Computes the LSTM cell backward propagation for 1 timestep.
robots: noindex

# tf.raw_ops.LSTMBlockCellGrad

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the LSTM cell backward propagation for 1 timestep.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.LSTMBlockCellGrad`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.LSTMBlockCellGrad(
    x,
    cs_prev,
    h_prev,
    w,
    wci,
    wcf,
    wco,
    b,
    i,
    cs,
    f,
    o,
    ci,
    co,
    cs_grad,
    h_grad,
    use_peephole,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This implementation is to be used in conjunction of LSTMBlockCell.

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
The previous cell state.
</td>
</tr><tr>
<td>
`h_prev`<a id="h_prev"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The previous h state.
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
`i`<a id="i"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The input gate.
</td>
</tr><tr>
<td>
`cs`<a id="cs"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
The cell state before the tanh.
</td>
</tr><tr>
<td>
`f`<a id="f"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The forget gate.
</td>
</tr><tr>
<td>
`o`<a id="o"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The output gate.
</td>
</tr><tr>
<td>
`ci`<a id="ci"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The cell input.
</td>
</tr><tr>
<td>
`co`<a id="co"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`. The cell after the tanh.
</td>
</tr><tr>
<td>
`cs_grad`<a id="cs_grad"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
The current gradient of cs.
</td>
</tr><tr>
<td>
`h_grad`<a id="h_grad"></a>
</td>
<td>
A `Tensor`. Must have the same type as `x`.
The gradient of h vector.
</td>
</tr><tr>
<td>
`use_peephole`<a id="use_peephole"></a>
</td>
<td>
A `bool`. Whether the cell uses peephole connections.
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
A tuple of `Tensor` objects (cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad).
</td>
</tr>
<tr>
<td>
`cs_prev_grad`<a id="cs_prev_grad"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`dicfo`<a id="dicfo"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`wci_grad`<a id="wci_grad"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`wcf_grad`<a id="wcf_grad"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr><tr>
<td>
`wco_grad`<a id="wco_grad"></a>
</td>
<td>
A `Tensor`. Has the same type as `x`.
</td>
</tr>
</table>

