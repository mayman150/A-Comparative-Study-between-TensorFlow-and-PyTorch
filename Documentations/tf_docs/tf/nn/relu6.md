description: Computes Rectified Linear 6: min(max(features, 0), 6).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.relu6" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.relu6

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Computes Rectified Linear 6: `min(max(features, 0), 6)`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.relu6(
    features, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

In comparison with <a href="../../tf/nn/relu.md"><code>tf.nn.relu</code></a>, relu6 activation functions have shown to
empirically perform better under low-precision conditions (e.g. fixed point
inference) by encouraging the model to learn sparse features earlier.
Source: [Convolutional Deep Belief Networks on CIFAR-10: Krizhevsky et al.,
2010](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf).

#### For example:



```
>>> x = tf.constant([-3.0, -1.0, 0.0, 6.0, 10.0], dtype=tf.float32)
>>> y = tf.nn.relu6(x)
>>> y.numpy()
array([0., 0., 0., 6., 6.], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`features`<a id="features"></a>
</td>
<td>
A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
`int16`, or `int8`.
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
A `Tensor` with the same type as `features`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
Convolutional Deep Belief Networks on CIFAR-10:
Krizhevsky et al., 2010
([pdf](http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf))
</td>
</tr>

</table>

