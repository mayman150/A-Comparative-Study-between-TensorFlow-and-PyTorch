description: Long short-term memory unit (LSTM) recurrent network cell.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.rnn_cell.LSTMCell" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="zero_state"/>
</div>

# tf.compat.v1.nn.rnn_cell.LSTMCell

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/legacy_cells.py#L857-L1183">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Long short-term memory unit (LSTM) recurrent network cell.

Inherits From: [`RNNCell`](../../../../../tf/compat/v1/nn/rnn_cell/RNNCell.md), [`Layer`](../../../../../tf/compat/v1/layers/Layer.md), [`Layer`](../../../../../tf/keras/layers/Layer.md), [`Module`](../../../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.rnn_cell.LSTMCell(
    num_units,
    use_peepholes=False,
    cell_clip=None,
    initializer=None,
    num_proj=None,
    proj_clip=None,
    num_unit_shards=None,
    num_proj_shards=None,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None,
    name=None,
    dtype=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The default non-peephole implementation is based on (Gers et al., 1999).
The peephole implementation is based on (Sak et al., 2014).

The class uses optional peep-hole connections, optional cell clipping, and
an optional projection layer.

Note that this cell is not optimized for performance. Please use
`tf.compat.v1.keras.layers.CuDNNLSTM` for better performance on GPU, or
<a href="../../../../../tf/raw_ops/LSTMBlockCell.md"><code>tf.raw_ops.LSTMBlockCell</code></a> for better performance on CPU.
References:
  Long short-term memory recurrent neural network architectures for large
  scale acoustic modeling:
    [Sak et al., 2014]
    (https://www.isca-speech.org/archive/interspeech_2014/i14_0338.html)
    ([pdf]
    (https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_0338.pdf))
  Learning to forget:
    [Gers et al., 1999]
    (http://digital-library.theiet.org/content/conferences/10.1049/cp_19991218)
    ([pdf](https://arxiv.org/pdf/1409.2329.pdf))
  Long Short-Term Memory:
    [Hochreiter et al., 1997]
    (https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
    ([pdf](http://ml.jku.at/publications/older/3504.pdf))

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_units`<a id="num_units"></a>
</td>
<td>
int, The number of units in the LSTM cell.
</td>
</tr><tr>
<td>
`use_peepholes`<a id="use_peepholes"></a>
</td>
<td>
bool, set True to enable diagonal/peephole connections.
</td>
</tr><tr>
<td>
`cell_clip`<a id="cell_clip"></a>
</td>
<td>
(optional) A float value, if provided the cell state is
clipped by this value prior to the cell output activation.
</td>
</tr><tr>
<td>
`initializer`<a id="initializer"></a>
</td>
<td>
(optional) The initializer to use for the weight and
projection matrices.
</td>
</tr><tr>
<td>
`num_proj`<a id="num_proj"></a>
</td>
<td>
(optional) int, The output dimensionality for the projection
matrices.  If None, no projection is performed.
</td>
</tr><tr>
<td>
`proj_clip`<a id="proj_clip"></a>
</td>
<td>
(optional) A float value.  If `num_proj > 0` and
`proj_clip` is provided, then the projected values are clipped
elementwise to within `[-proj_clip, proj_clip]`.
</td>
</tr><tr>
<td>
`num_unit_shards`<a id="num_unit_shards"></a>
</td>
<td>
Deprecated, will be removed by Jan. 2017. Use a
variable_scope partitioner instead.
</td>
</tr><tr>
<td>
`num_proj_shards`<a id="num_proj_shards"></a>
</td>
<td>
Deprecated, will be removed by Jan. 2017. Use a
variable_scope partitioner instead.
</td>
</tr><tr>
<td>
`forget_bias`<a id="forget_bias"></a>
</td>
<td>
Biases of the forget gate are initialized by default to 1
in order to reduce the scale of forgetting at the beginning of the
training. Must set it manually to `0.0` when restoring from
CudnnLSTM trained checkpoints.
</td>
</tr><tr>
<td>
`state_is_tuple`<a id="state_is_tuple"></a>
</td>
<td>
If True, accepted and returned states are 2-tuples of
the `c_state` and `m_state`.  If False, they are concatenated along
the column axis.  This latter behavior will soon be deprecated.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
Activation function of the inner states.  Default: `tanh`.
It could also be string that is within Keras activation function
names.
</td>
</tr><tr>
<td>
`reuse`<a id="reuse"></a>
</td>
<td>
(optional) Python boolean describing whether to reuse variables
in an existing scope.  If not `True`, and the existing scope already
has the given variables, an error is raised.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
String, the name of the layer. Layers with the same name will
share weights, but to avoid mistakes we require reuse=True in such
cases.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Default dtype of the layer (default of `None` means use the
type of the first input). Required when `build` is called before
`call`.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Dict, keyword named properties for common layer attributes,
like `trainable` etc when constructing the cell from configs of
get_config().  When restoring from CudnnLSTM-trained checkpoints,
use `CudnnCompatibleLSTMCell` instead.
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/legacy_cells.py#L292-L328">View source</a>

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





