description: Cell class for the GRU layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.GRUCell" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_dropout_mask_for_cell"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="get_recurrent_dropout_mask_for_cell"/>
<meta itemprop="property" content="reset_dropout_mask"/>
<meta itemprop="property" content="reset_recurrent_dropout_mask"/>
</div>

# tf.keras.layers.GRUCell

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/gru.py#L45-L391">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Cell class for the GRU layer.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.GRUCell(
    units,
    activation=&#x27;tanh&#x27;,
    recurrent_activation=&#x27;sigmoid&#x27;,
    use_bias=True,
    kernel_initializer=&#x27;glorot_uniform&#x27;,
    recurrent_initializer=&#x27;orthogonal&#x27;,
    bias_initializer=&#x27;zeros&#x27;,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    reset_after=True,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
for details about the usage of RNN API.

This class processes one step within the whole time sequence input, whereas
`tf.keras.layer.GRU` processes the whole sequence.

#### For example:



```
>>> inputs = tf.random.normal([32, 10, 8])
>>> rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4))
>>> output = rnn(inputs)
>>> print(output.shape)
(32, 4)
>>> rnn = tf.keras.layers.RNN(
...    tf.keras.layers.GRUCell(4),
...    return_sequences=True,
...    return_state=True)
>>> whole_sequence_output, final_state = rnn(inputs)
>>> print(whole_sequence_output.shape)
(32, 10, 4)
>>> print(final_state.shape)
(32, 4)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`units`<a id="units"></a>
</td>
<td>
Positive integer, dimensionality of the output space.
</td>
</tr><tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
Activation function to use. Default: hyperbolic tangent
(`tanh`). If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
</td>
</tr><tr>
<td>
`recurrent_activation`<a id="recurrent_activation"></a>
</td>
<td>
Activation function to use for the recurrent step.
Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
applied (ie. "linear" activation: `a(x) = x`).
</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>
Boolean, (default `True`), whether the layer uses a bias vector.
</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>
Initializer for the `kernel` weights matrix,
used for the linear transformation of the inputs. Default:
`glorot_uniform`.
</td>
</tr><tr>
<td>
`recurrent_initializer`<a id="recurrent_initializer"></a>
</td>
<td>
Initializer for the `recurrent_kernel`
weights matrix, used for the linear transformation of the recurrent
state.  Default: `orthogonal`.
</td>
</tr><tr>
<td>
`bias_initializer`<a id="bias_initializer"></a>
</td>
<td>
Initializer for the bias vector. Default: `zeros`.
</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>
Regularizer function applied to the `kernel` weights
matrix. Default: `None`.
</td>
</tr><tr>
<td>
`recurrent_regularizer`<a id="recurrent_regularizer"></a>
</td>
<td>
Regularizer function applied to the
`recurrent_kernel` weights matrix. Default: `None`.
</td>
</tr><tr>
<td>
`bias_regularizer`<a id="bias_regularizer"></a>
</td>
<td>
Regularizer function applied to the bias vector.
Default: `None`.
</td>
</tr><tr>
<td>
`kernel_constraint`<a id="kernel_constraint"></a>
</td>
<td>
Constraint function applied to the `kernel` weights
matrix. Default: `None`.
</td>
</tr><tr>
<td>
`recurrent_constraint`<a id="recurrent_constraint"></a>
</td>
<td>
Constraint function applied to the
`recurrent_kernel` weights matrix. Default: `None`.
</td>
</tr><tr>
<td>
`bias_constraint`<a id="bias_constraint"></a>
</td>
<td>
Constraint function applied to the bias vector. Default:
`None`.
</td>
</tr><tr>
<td>
`dropout`<a id="dropout"></a>
</td>
<td>
Float between 0 and 1. Fraction of the units to drop for the
linear transformation of the inputs. Default: 0.
</td>
</tr><tr>
<td>
`recurrent_dropout`<a id="recurrent_dropout"></a>
</td>
<td>
Float between 0 and 1. Fraction of the units to drop
for the linear transformation of the recurrent state. Default: 0.
</td>
</tr><tr>
<td>
`reset_after`<a id="reset_after"></a>
</td>
<td>
GRU convention (whether to apply reset gate after or
before matrix multiplication). False = "before",
True = "after" (default and cuDNN compatible).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call arguments</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A 2D tensor, with shape of `[batch, feature]`.
</td>
</tr><tr>
<td>
`states`<a id="states"></a>
</td>
<td>
A 2D tensor with shape of `[batch, units]`, which is the state
from the previous time step. For timestep 0, the initial state provided
by user will be feed to cell.
</td>
</tr><tr>
<td>
`training`<a id="training"></a>
</td>
<td>
Python boolean indicating whether the layer should behave in
training mode or in inference mode. Only relevant when `dropout` or
`recurrent_dropout` is used.
</td>
</tr>
</table>



## Methods

<h3 id="get_dropout_mask_for_cell"><code>get_dropout_mask_for_cell</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/dropout_rnn_cell_mixin.py#L110-L129">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_dropout_mask_for_cell(
    inputs, training, count=1
)
</code></pre>

Get the dropout mask for RNN cell's input.

It will create mask based on context if there isn't any existing cached
mask. If a new mask is generated, it will update the cache in the cell.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
The input tensor whose shape will be used to generate dropout
mask.
</td>
</tr><tr>
<td>
`training`
</td>
<td>
Boolean tensor, whether its in training mode, dropout will
be ignored in non-training mode.
</td>
</tr><tr>
<td>
`count`
</td>
<td>
Int, how many dropout mask will be generated. It is useful for
cell that has internal weights fused together.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of mask tensor, generated or cached mask based on context.
</td>
</tr>

</table>



<h3 id="get_initial_state"><code>get_initial_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/gru.py#L388-L391">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_initial_state(
    inputs=None, batch_size=None, dtype=None
)
</code></pre>




<h3 id="get_recurrent_dropout_mask_for_cell"><code>get_recurrent_dropout_mask_for_cell</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/dropout_rnn_cell_mixin.py#L131-L150">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_recurrent_dropout_mask_for_cell(
    inputs, training, count=1
)
</code></pre>

Get the recurrent dropout mask for RNN cell.

It will create mask based on context if there isn't any existing cached
mask. If a new mask is generated, it will update the cache in the cell.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
The input tensor whose shape will be used to generate dropout
mask.
</td>
</tr><tr>
<td>
`training`
</td>
<td>
Boolean tensor, whether its in training mode, dropout will
be ignored in non-training mode.
</td>
</tr><tr>
<td>
`count`
</td>
<td>
Int, how many dropout mask will be generated. It is useful for
cell that has internal weights fused together.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of mask tensor, generated or cached mask based on context.
</td>
</tr>

</table>



<h3 id="reset_dropout_mask"><code>reset_dropout_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/dropout_rnn_cell_mixin.py#L70-L79">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_dropout_mask()
</code></pre>

Reset the cached dropout masks if any.

This is important for the RNN layer to invoke this in it `call()` method
so that the cached mask is cleared before calling the `cell.call()`. The
mask should be cached across the timestep within the same batch, but
shouldn't be cached between batches. Otherwise it will introduce
unreasonable bias against certain index of data within the batch.

<h3 id="reset_recurrent_dropout_mask"><code>reset_recurrent_dropout_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/dropout_rnn_cell_mixin.py#L81-L90">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_recurrent_dropout_mask()
</code></pre>

Reset the cached recurrent dropout masks if any.

This is important for the RNN layer to invoke this in it call() method
so that the cached mask is cleared before calling the cell.call(). The
mask should be cached across the timestep within the same batch, but
shouldn't be cached between batches. Otherwise it will introduce
unreasonable bias against certain index of data within the batch.



