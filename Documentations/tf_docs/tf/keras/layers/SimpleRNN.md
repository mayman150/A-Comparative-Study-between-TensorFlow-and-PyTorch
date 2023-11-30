description: Fully-connected RNN where the output is to be fed back to input.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.SimpleRNN" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="reset_states"/>
</div>

# tf.keras.layers.SimpleRNN

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/simple_rnn.py#L253-L510">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Fully-connected RNN where the output is to be fed back to input.

Inherits From: [`RNN`](../../../tf/keras/layers/RNN.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.SimpleRNN(
    units,
    activation=&#x27;tanh&#x27;,
    use_bias=True,
    kernel_initializer=&#x27;glorot_uniform&#x27;,
    recurrent_initializer=&#x27;orthogonal&#x27;,
    bias_initializer=&#x27;zeros&#x27;,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
for details about the usage of RNN API.

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
Activation function to use.
Default: hyperbolic tangent (`tanh`).
If you pass None, no activation is applied
(ie. "linear" activation: `a(x) = x`).
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
`activity_regularizer`<a id="activity_regularizer"></a>
</td>
<td>
Regularizer function applied to the output of the
layer (its "activation"). Default: `None`.
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
`recurrent_kernel` weights matrix.  Default: `None`.
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
Float between 0 and 1.
Fraction of the units to drop for the linear transformation of the
inputs. Default: 0.
</td>
</tr><tr>
<td>
`recurrent_dropout`<a id="recurrent_dropout"></a>
</td>
<td>
Float between 0 and 1.
Fraction of the units to drop for the linear transformation of the
recurrent state. Default: 0.
</td>
</tr><tr>
<td>
`return_sequences`<a id="return_sequences"></a>
</td>
<td>
Boolean. Whether to return the last output
in the output sequence, or the full sequence. Default: `False`.
</td>
</tr><tr>
<td>
`return_state`<a id="return_state"></a>
</td>
<td>
Boolean. Whether to return the last state
in addition to the output. Default: `False`
</td>
</tr><tr>
<td>
`go_backwards`<a id="go_backwards"></a>
</td>
<td>
Boolean (default False).
If True, process the input sequence backwards and return the
reversed sequence.
</td>
</tr><tr>
<td>
`stateful`<a id="stateful"></a>
</td>
<td>
Boolean (default False). If True, the last state
for each sample at index i in a batch will be used as initial
state for the sample of index i in the following batch.
</td>
</tr><tr>
<td>
`unroll`<a id="unroll"></a>
</td>
<td>
Boolean (default False).
If True, the network will be unrolled,
else a symbolic loop will be used.
Unrolling can speed-up a RNN,
although it tends to be more memory-intensive.
Unrolling is only suitable for short sequences.
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
A 3D tensor, with shape `[batch, timesteps, feature]`.
</td>
</tr><tr>
<td>
`mask`<a id="mask"></a>
</td>
<td>
Binary tensor of shape `[batch, timesteps]` indicating whether
a given timestep should be masked. An individual `True` entry indicates
that the corresponding timestep should be utilized, while a `False`
entry indicates that the corresponding timestep should be ignored.
</td>
</tr><tr>
<td>
`training`<a id="training"></a>
</td>
<td>
Python boolean indicating whether the layer should behave in
training mode or in inference mode. This argument is passed to the cell
when calling it. This is only relevant if `dropout` or
`recurrent_dropout` is used.
</td>
</tr><tr>
<td>
`initial_state`<a id="initial_state"></a>
</td>
<td>
List of initial state tensors to be passed to the first
call of the cell.
</td>
</tr>
</table>



#### Examples:



```python
inputs = np.random.random([32, 10, 8]).astype(np.float32)
simple_rnn = tf.keras.layers.SimpleRNN(4)

output = simple_rnn(inputs)  # The output has shape `[32, 4]`.

simple_rnn = tf.keras.layers.SimpleRNN(
    4, return_sequences=True, return_state=True)

# whole_sequence_output has shape `[32, 10, 4]`.
# final_state has shape `[32, 4]`.
whole_sequence_output, final_state = simple_rnn(inputs)
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`bias_constraint`<a id="bias_constraint"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`bias_initializer`<a id="bias_initializer"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`bias_regularizer`<a id="bias_regularizer"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`dropout`<a id="dropout"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`kernel_constraint`<a id="kernel_constraint"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`kernel_initializer`<a id="kernel_initializer"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`kernel_regularizer`<a id="kernel_regularizer"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`recurrent_constraint`<a id="recurrent_constraint"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`recurrent_dropout`<a id="recurrent_dropout"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`recurrent_initializer`<a id="recurrent_initializer"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`recurrent_regularizer`<a id="recurrent_regularizer"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`states`<a id="states"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`units`<a id="units"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`use_bias`<a id="use_bias"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="reset_states"><code>reset_states</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/base_rnn.py#L846-L945">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_states(
    states=None
)
</code></pre>

Reset the recorded states for the stateful RNN layer.

Can only be used when RNN layer is constructed with `stateful` = `True`.
Args:
  states: Numpy arrays that contains the value for the initial state,
    which will be feed to cell at the first time step. When the value is
    None, zero filled numpy array will be created based on the cell
    state size.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`AttributeError`
</td>
<td>
When the RNN layer is not stateful.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
When the batch size of the RNN layer is unknown.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
When the input numpy array is not compatible with the RNN
layer state, either size wise or dtype wise.
</td>
</tr>
</table>





