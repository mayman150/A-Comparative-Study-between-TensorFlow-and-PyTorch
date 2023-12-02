<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.rnn.EmbeddingWrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="graph"/>
<meta itemprop="property" content="output_size"/>
<meta itemprop="property" content="scope_name"/>
<meta itemprop="property" content="state_size"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="zero_state"/>
</div>

# tf.contrib.rnn.EmbeddingWrapper

## Class `EmbeddingWrapper`

Operator adding input embedding to the given cell.

Inherits From: [`RNNCell`](../../../tf/nn/rnn_cell/RNNCell.md)

<!-- Placeholder for "Used in" -->

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your inputs in time,
do the embedding on this batch-concatenated sequence, then split it and
feed into your RNN.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    cell,
    embedding_classes,
    embedding_size,
    initializer=None,
    reuse=None
)
```

Create a cell with an added input embedding.


#### Args:


* <b>`cell`</b>: an RNNCell, an embedding will be put before its inputs.
* <b>`embedding_classes`</b>: integer, how many symbols will be embedded.
* <b>`embedding_size`</b>: integer, the size of the vectors we embed into.
* <b>`initializer`</b>: an initializer to use when creating the embedding;
  if None, the initializer from variable scope or a default one is used.
* <b>`reuse`</b>: (optional) Python boolean describing whether to reuse variables
  in an existing scope.  If not `True`, and the existing scope already has
  the given variables, an error is raised.


#### Raises:


* <b>`TypeError`</b>: if cell is not an RNNCell.
* <b>`ValueError`</b>: if embedding_classes is not positive.



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




