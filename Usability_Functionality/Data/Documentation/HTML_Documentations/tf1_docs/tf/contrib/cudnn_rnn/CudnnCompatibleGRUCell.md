<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="graph"/>
<meta itemprop="property" content="output_size"/>
<meta itemprop="property" content="scope_name"/>
<meta itemprop="property" content="state_size"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_initial_state"/>
<meta itemprop="property" content="zero_state"/>
</div>

# tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell

## Class `CudnnCompatibleGRUCell`

Cudnn Compatible GRUCell.

Inherits From: [`GRUCell`](../../../tf/nn/rnn_cell/GRUCell.md)

<!-- Placeholder for "Used in" -->

A GRU impl akin to <a href="../../../tf/nn/rnn_cell/GRUCell.md"><code>tf.compat.v1.nn.rnn_cell.GRUCell</code></a> to use along with
<a href="../../../tf/contrib/cudnn_rnn/CudnnGRU.md"><code>tf.contrib.cudnn_rnn.CudnnGRU</code></a>. The latter's params can be used by
it seamlessly.

It differs from platform-independent GRUs in how the new memory gate is
calculated. Nvidia picks this variant based on GRU author's[1] suggestion and
the fact it has no accuracy impact[2].
[1] https://arxiv.org/abs/1406.1078
[2] http://svail.github.io/diff_graphs/

Cudnn compatible GRU (from Cudnn library user guide):
```python
# reset gate
$$r_t = \sigma(x_t * W_r + h_t-1 * R_h + b_{Wr} + b_{Rr})$$
# update gate
$$u_t = \sigma(x_t * W_u + h_t-1 * R_u + b_{Wu} + b_{Ru})$$
# new memory gate
$$h'_t = tanh(x_t * W_h + r_t .* (h_t-1 * R_h + b_{Rh}) + b_{Wh})$$
$$h_t = (1 - u_t) .* h'_t + u_t .* h_t-1$$
```

Other GRU (see <a href="../../../tf/nn/rnn_cell/GRUCell.md"><code>tf.compat.v1.nn.rnn_cell.GRUCell</code></a> and
<a href="../../../tf/contrib/rnn/GRUBlockCell.md"><code>tf.contrib.rnn.GRUBlockCell</code></a>):
```python
# new memory gate
\\(h'_t = tanh(x_t * W_h + (r_t .* h_t-1) * R_h + b_{Wh})\\)
```
which is not equivalent to Cudnn GRU: in addition to the extra bias term b_Rh,
```python
\\(r .* (h * R) != (r .* h) * R\\)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_units,
    reuse=None,
    kernel_initializer=None
)
```

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This class is equivalent as tf.keras.layers.GRUCell, and will be replaced by that in Tensorflow 2.0.



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




