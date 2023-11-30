description: Operator adding dropout to inputs and outputs of the given cell.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.rnn_cell.DropoutWrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="zero_state"/>
</div>

# tf.compat.v1.nn.rnn_cell.DropoutWrapper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/legacy_cell_wrappers.py#L186-L531">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Operator adding dropout to inputs and outputs of the given cell.

Inherits From: [`RNNCell`](../../../../../tf/compat/v1/nn/rnn_cell/RNNCell.md), [`Layer`](../../../../../tf/compat/v1/layers/Layer.md), [`Layer`](../../../../../tf/keras/layers/Layer.md), [`Module`](../../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.rnn_cell.DropoutWrapper(
    cell,
    input_keep_prob=1.0,
    output_keep_prob=1.0,
    state_keep_prob=1.0,
    variational_recurrent=False,
    input_size=None,
    dtype=None,
    seed=None,
    dropout_state_filter_visitor=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cell`<a id="cell"></a>
</td>
<td>
an RNNCell, a projection to output_size is added to it.
</td>
</tr><tr>
<td>
`input_keep_prob`<a id="input_keep_prob"></a>
</td>
<td>
unit Tensor or float between 0 and 1, input keep
probability; if it is constant and 1, no input dropout will be
added.
</td>
</tr><tr>
<td>
`output_keep_prob`<a id="output_keep_prob"></a>
</td>
<td>
unit Tensor or float between 0 and 1, output keep
probability; if it is constant and 1, no output dropout will be
added.
</td>
</tr><tr>
<td>
`state_keep_prob`<a id="state_keep_prob"></a>
</td>
<td>
unit Tensor or float between 0 and 1, output keep
probability; if it is constant and 1, no output dropout will be
added. State dropout is performed on the outgoing states of the
cell. **Note** the state components to which dropout is applied when
`state_keep_prob` is in `(0, 1)` are also determined by the argument
`dropout_state_filter_visitor` (e.g. by default dropout is never
applied to the `c` component of an `LSTMStateTuple`).
</td>
</tr><tr>
<td>
`variational_recurrent`<a id="variational_recurrent"></a>
</td>
<td>
Python bool.  If `True`, then the same dropout
pattern is applied across all time steps per run call. If this
parameter is set, `input_size` **must** be provided.
</td>
</tr><tr>
<td>
`input_size`<a id="input_size"></a>
</td>
<td>
(optional) (possibly nested tuple of) `TensorShape`
objects containing the depth(s) of the input tensors expected to be
passed in to the `DropoutWrapper`.  Required and used **iff**
`variational_recurrent = True` and `input_keep_prob < 1`.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
(optional) The `dtype` of the input, state, and output tensors.
Required and used **iff** `variational_recurrent = True`.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
(optional) integer, the randomness seed.
</td>
</tr><tr>
<td>
`dropout_state_filter_visitor`<a id="dropout_state_filter_visitor"></a>
</td>
<td>
(optional), default: (see below).
Function that takes any hierarchical level of the state and returns
a scalar or depth=1 structure of Python booleans describing which
terms in the state should be dropped out.  In addition, if the
function returns `True`, dropout is applied across this sublevel.
If the function returns `False`, dropout is not applied across this
entire sublevel.  Default behavior: perform dropout on all terms
except the memory (`c`) state of `LSTMCellState` objects, and don't
try to apply dropout to `TensorArray` objects:
```
def dropout_state_filter_visitor(s):
  # Never perform dropout on the c state.
  if isinstance(s, LSTMCellState):
    return LSTMCellState(c=False, h=True)
  elif isinstance(s, TensorArray):
    return False
  return True
```
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
dict of keyword arguments for base layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
if `cell` is not an `RNNCell`, or `keep_state_fn` is
provided but not `callable`.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if any of the keep_probs are not between 0 and 1.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`graph`<a id="graph"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`output_size`<a id="output_size"></a>
</td>
<td>
Integer or TensorShape: size of outputs produced by this cell.
</td>
</tr><tr>
<td>
`scope_name`<a id="scope_name"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`state_size`<a id="state_size"></a>
</td>
<td>
size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of
Integers or TensorShapes.
</td>
</tr><tr>
<td>
`wrapped_cell`<a id="wrapped_cell"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="apply"><code>apply</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/legacy_tf_layers/base.py#L239-L240">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    *args, **kwargs
)
</code></pre>




<h3 id="get_initial_state"><code>get_initial_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/legacy_cells.py#L254-L290">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_initial_state(
    inputs=None, batch_size=None, dtype=None
)
</code></pre>




<h3 id="get_losses_for"><code>get_losses_for</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/base_layer_v1.py#L1467-L1484">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_losses_for(
    inputs
)
</code></pre>

Retrieves losses relevant to a specific set of inputs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor or list/tuple of input tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of loss tensors of the layer that depend on `inputs`.
</td>
</tr>

</table>



<h3 id="get_updates_for"><code>get_updates_for</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/base_layer_v1.py#L1448-L1465">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_updates_for(
    inputs
)
</code></pre>

Retrieves updates relevant to a specific set of inputs.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Input tensor or list/tuple of input tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of update ops of the layer that depend on `inputs`.
</td>
</tr>

</table>



<h3 id="zero_state"><code>zero_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/legacy_cell_wrappers.py#L157-L159">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>zero_state(
    batch_size, dtype
)
</code></pre>

Return zero-filled state tensor(s).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch_size`
</td>
<td>
int, float, or unit Tensor representing the batch size.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
the data type to use for the state.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
If `state_size` is an int or TensorShape, then the return value is a
`N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

If `state_size` is a nested list or tuple, then the return value is
a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size, s]` for each s in `state_size`.
</td>
</tr>

</table>





