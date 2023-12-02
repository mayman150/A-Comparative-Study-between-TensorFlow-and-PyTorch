<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.rnn.UGRNNCell" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="graph"/>
<meta itemprop="property" content="output_size"/>
<meta itemprop="property" content="scope_name"/>
<meta itemprop="property" content="state_size"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="zero_state"/>
</div>

# tf.contrib.rnn.UGRNNCell

## Class `UGRNNCell`

Update Gate Recurrent Neural Network (UGRNN) cell.

Inherits From: [`RNNCell`](../../../tf/nn/rnn_cell/RNNCell.md)

<!-- Placeholder for "Used in" -->

Compromise between a LSTM/GRU and a vanilla RNN.  There is only one
gate, and that is to determine whether the unit should be
integrating or computing instantaneously.  This is the recurrent
idea of the feedforward Highway Network.

This implements the recurrent cell from the paper:

  https://arxiv.org/abs/1611.09913

Jasmine Collins, Jascha Sohl-Dickstein, and David Sussillo.
"Capacity and Trainability in Recurrent Neural Networks" Proc. ICLR 2017.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_units,
    initializer=None,
    forget_bias=1.0,
    activation=tf.math.tanh,
    reuse=None
)
```

Initialize the parameters for an UGRNN cell.


#### Args:


* <b>`num_units`</b>: int, The number of units in the UGRNN cell
* <b>`initializer`</b>: (optional) The initializer to use for the weight matrices.
* <b>`forget_bias`</b>: (optional) float, default 1.0, The initial bias of the
  forget gate, used to reduce the scale of forgetting at the beginning
  of the training.
* <b>`activation`</b>: (optional) Activation function of the inner states.
  Default is <a href="../../../tf/math/tanh.md"><code>tf.tanh</code></a>.
* <b>`reuse`</b>: (optional) Python boolean describing whether to reuse variables
  in an existing scope.  If not `True`, and the existing scope already has
  the given variables, an error is raised.



## Properties

<h3 id="graph"><code>graph</code></h3>

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Stop using this property because tf.layers layers no longer track their graph.

<h3 id="output_size"><code>output_size</code></h3>

Integer or TensorShape: size of outputs produced by this cell.


<h3 id="scope_name"><code>scope_name</code></h3>




<h3 id="state_size"><code>state_size</code></h3>

size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of Integers
or TensorShapes.



## Methods

<h3 id="get_initial_state"><code>get_initial_state</code></h3>

``` python
get_initial_state(
    inputs=None,
    batch_size=None,
    dtype=None
)
```




<h3 id="zero_state"><code>zero_state</code></h3>

``` python
zero_state(
    batch_size,
    dtype
)
```

Return zero-filled state tensor(s).


#### Args:


* <b>`batch_size`</b>: int, float, or unit Tensor representing the batch size.
* <b>`dtype`</b>: the data type to use for the state.


#### Returns:

If `state_size` is an int or TensorShape, then the return value is a
`N-D` tensor of shape `[batch_size, state_size]` filled with zeros.

If `state_size` is a nested list or tuple, then the return value is
a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size, s]` for each s in `state_size`.




