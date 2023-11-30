description: Group normalization layer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.GroupNormalization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.GroupNormalization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/normalization/group_normalization.py#L31-L269">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Group normalization layer.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.GroupNormalization(
    groups=32,
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer=&#x27;zeros&#x27;,
    gamma_initializer=&#x27;ones&#x27;,
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Group Normalization divides the channels into groups and computes
within each group the mean and variance for normalization.
Empirically, its accuracy is more stable than batch norm in a wide
range of small batch sizes, if learning rate is adjusted linearly
with batch sizes.

Relation to Layer Normalization:
If the number of groups is set to 1, then this operation becomes nearly
identical to Layer Normalization (see Layer Normalization docs for details).

Relation to Instance Normalization:
If the number of groups is set to the input dimension (number of groups is
equal to number of channels), then this operation becomes identical to
Instance Normalization.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`groups`<a id="groups"></a>
</td>
<td>
Integer, the number of groups for Group Normalization. Can be in
the range [1, N] where N is the input dimension. The input dimension
must be divisible by the number of groups. Defaults to `32`.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
Integer or List/Tuple. The axis or axes to normalize across.
Typically, this is the features axis/axes. The left-out axes are
typically the batch axis/axes. `-1` is the last dimension in the
input. Defaults to `-1`.
</td>
</tr><tr>
<td>
`epsilon`<a id="epsilon"></a>
</td>
<td>
Small float added to variance to avoid dividing by zero. Defaults
to 1e-3
</td>
</tr><tr>
<td>
`center`<a id="center"></a>
</td>
<td>
If True, add offset of `beta` to normalized tensor. If False,
`beta` is ignored. Defaults to `True`.
</td>
</tr><tr>
<td>
`scale`<a id="scale"></a>
</td>
<td>
If True, multiply by `gamma`. If False, `gamma` is not used.
When the next layer is linear (also e.g. <a href="../../../tf/nn/relu.md"><code>nn.relu</code></a>), this can be
disabled since the scaling will be done by the next layer.
Defaults to `True`.
</td>
</tr><tr>
<td>
`beta_initializer`<a id="beta_initializer"></a>
</td>
<td>
Initializer for the beta weight. Defaults to zeros.
</td>
</tr><tr>
<td>
`gamma_initializer`<a id="gamma_initializer"></a>
</td>
<td>
Initializer for the gamma weight. Defaults to ones.
</td>
</tr><tr>
<td>
`beta_regularizer`<a id="beta_regularizer"></a>
</td>
<td>
Optional regularizer for the beta weight. None by
default.
</td>
</tr><tr>
<td>
`gamma_regularizer`<a id="gamma_regularizer"></a>
</td>
<td>
Optional regularizer for the gamma weight. None by
default.
</td>
</tr><tr>
<td>
`beta_constraint`<a id="beta_constraint"></a>
</td>
<td>
Optional constraint for the beta weight. None by default.
</td>
</tr><tr>
<td>
`gamma_constraint`<a id="gamma_constraint"></a>
</td>
<td>
Optional constraint for the gamma weight. None by
default.  Input shape: Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis) when using this
layer as the first layer in a model.  Output shape: Same shape as input.
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
Input tensor (of any rank).
</td>
</tr><tr>
<td>
`mask`<a id="mask"></a>
</td>
<td>
The mask parameter is a tensor that indicates the weight for each
position in the input tensor when computing the mean and variance.
</td>
</tr>
</table>


Reference: - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)

