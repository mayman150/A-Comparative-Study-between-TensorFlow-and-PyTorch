description: Performs spectral normalization on the weights of a target layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.SpectralNormalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="normalize_weights"/>
</div>

# tf.keras.layers.SpectralNormalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/normalization/spectral_normalization.py#L26-L141">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Performs spectral normalization on the weights of a target layer.

Inherits From: [`Wrapper`](../../../tf/keras/layers/Wrapper.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.SpectralNormalization(
    layer, power_iterations=1, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This wrapper controls the Lipschitz constant of the weights of a layer by
constraining their spectral norm, which can stabilize the training of GANs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layer`<a id="layer"></a>
</td>
<td>
A <a href="../../../tf/keras/layers/Layer.md"><code>keras.layers.Layer</code></a> instance that
has either a `kernel` (e.g. `Conv2D`, `Dense`...)
or an `embeddings` attribute (`Embedding` layer).
</td>
</tr><tr>
<td>
`power_iterations`<a id="power_iterations"></a>
</td>
<td>
int, the number of iterations during normalization.
</td>
</tr>
</table>



#### Examples:



Wrap <a href="../../../tf/keras/layers/Conv2D.md"><code>keras.layers.Conv2D</code></a>:
```
>>> x = np.random.rand(1, 10, 10, 1)
>>> conv2d = SpectralNormalization(tf.keras.layers.Conv2D(2, 2))
>>> y = conv2d(x)
>>> y.shape
TensorShape([1, 9, 9, 2])
```

Wrap <a href="../../../tf/keras/layers/Dense.md"><code>keras.layers.Dense</code></a>:
```
>>> x = np.random.rand(1, 10, 10, 1)
>>> dense = SpectralNormalization(tf.keras.layers.Dense(10))
>>> y = dense(x)
>>> y.shape
TensorShape([1, 10, 10, 10])
```

#### Reference:



- [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).

## Methods

<h3 id="normalize_weights"><code>normalize_weights</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/normalization/spectral_normalization.py#L108-L136">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>normalize_weights()
</code></pre>

Generate spectral normalized weights.

This method will update the value of `self.kernel` with the
spectral normalized value, so that the layer is ready for `call()`.



