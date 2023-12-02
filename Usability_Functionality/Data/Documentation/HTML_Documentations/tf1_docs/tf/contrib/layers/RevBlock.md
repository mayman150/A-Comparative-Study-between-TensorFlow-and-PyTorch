<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.RevBlock" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="graph"/>
<meta itemprop="property" content="scope_name"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="backward"/>
<meta itemprop="property" content="forward"/>
</div>

# tf.contrib.layers.RevBlock

## Class `RevBlock`

Block of reversible layers. See rev_block.

Inherits From: [`Layer`](../../../tf/layers/Layer.md)

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    f,
    g,
    num_layers=1,
    f_side_input=None,
    g_side_input=None,
    use_efficient_backprop=True,
    name='revblock',
    **kwargs
)
```






## Properties

<h3 id="graph"><code>graph</code></h3>

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Stop using this property because tf.layers layers no longer track their graph.

<h3 id="scope_name"><code>scope_name</code></h3>






## Methods

<h3 id="backward"><code>backward</code></h3>

``` python
backward(
    y1,
    y2
)
```




<h3 id="forward"><code>forward</code></h3>

``` python
forward(
    x1,
    x2
)
```






