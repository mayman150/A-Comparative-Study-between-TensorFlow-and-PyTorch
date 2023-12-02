<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.rnn.LSTMBlockWrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="graph"/>
<meta itemprop="property" content="num_units"/>
<meta itemprop="property" content="scope_name"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf.contrib.rnn.LSTMBlockWrapper

## Class `LSTMBlockWrapper`

This is a helper class that provides housekeeping for LSTM cells.

Inherits From: [`Layer`](../../../tf/layers/Layer.md)

<!-- Placeholder for "Used in" -->

This may be useful for alternative LSTM and similar type of cells.
The subclasses must implement `_call_cell` method and `num_units` property.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    trainable=True,
    name=None,
    dtype=None,
    **kwargs
)
```






## Properties

<h3 id="graph"><code>graph</code></h3>

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Stop using this property because tf.layers layers no longer track their graph.

<h3 id="num_units"><code>num_units</code></h3>

Number of units in this cell (output dimension).


<h3 id="scope_name"><code>scope_name</code></h3>






