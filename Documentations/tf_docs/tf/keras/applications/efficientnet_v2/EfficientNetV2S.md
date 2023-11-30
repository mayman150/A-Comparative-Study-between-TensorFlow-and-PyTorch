description: Instantiates the EfficientNetV2S architecture.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.efficientnet_v2.EfficientNetV2S" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.efficientnet_v2.EfficientNetV2S

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/applications/efficientnet_v2.py#L1236-L1263">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Instantiates the EfficientNetV2S architecture.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.applications.EfficientNetV2S`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=True,
    weights=&#x27;imagenet&#x27;,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation=&#x27;softmax&#x27;,
    include_preprocessing=True
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Reference:


- [EfficientNetV2: Smaller Models and Faster Training](
    https://arxiv.org/abs/2104.00298) (ICML 2021)

This function returns a Keras image classification model,
optionally loaded with weights pre-trained on ImageNet.

For image classification use cases, see
[this page for detailed examples](
  https://keras.io/api/applications/#usage-examples-for-image-classification-models).

For transfer learning use cases, make sure to read the
[guide to transfer learning & fine-tuning](
  https://keras.io/guides/transfer_learning/).

Note: each Keras Application expects a specific kind of input preprocessing.
For EfficientNetV2, by default input preprocessing is included as a part of
the model (as a `Rescaling` layer), and thus
<a href="../../../../tf/keras/applications/efficientnet_v2/preprocess_input.md"><code>tf.keras.applications.efficientnet_v2.preprocess_input</code></a> is actually a
pass-through function. In this use case, EfficientNetV2 models expect their
inputs to be float tensors of pixels with values in the [0-255] range.
At the same time, preprocessing as a part of the model (i.e. `Rescaling`
layer) can be disabled by setting `include_preprocessing` argument to False.
With preprocessing disabled EfficientNetV2 models expect their inputs to be
float tensors of pixels with values in the [-1, 1] range.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`include_top`<a id="include_top"></a>
</td>
<td>
Boolean, whether to include the fully-connected
layer at the top of the network. Defaults to `True`.
</td>
</tr><tr>
<td>
`weights`<a id="weights"></a>
</td>
<td>
One of `None` (random initialization),
`"imagenet"` (pre-training on ImageNet),
or the path to the weights file to be loaded. Defaults to `"imagenet"`.
</td>
</tr><tr>
<td>
`input_tensor`<a id="input_tensor"></a>
</td>
<td>
Optional Keras tensor
(i.e. output of <a href="../../../../tf/keras/Input.md"><code>layers.Input()</code></a>)
to use as image input for the model.
</td>
</tr><tr>
<td>
`input_shape`<a id="input_shape"></a>
</td>
<td>
Optional shape tuple, only to be specified
if `include_top` is False.
It should have exactly 3 inputs channels.
</td>
</tr><tr>
<td>
`pooling`<a id="pooling"></a>
</td>
<td>
Optional pooling mode for feature extraction
when `include_top` is `False`.
- `None` means that the output of the model will be
    the 4D tensor output of the
    last convolutional layer.
- `"avg"` means that global average pooling
    will be applied to the output of the
    last convolutional layer, and thus
    the output of the model will be a 2D tensor.
- `"max"` means that global max pooling will
    be applied.
Defaults to `None`.
</td>
</tr><tr>
<td>
`classes`<a id="classes"></a>
</td>
<td>
Optional number of classes to classify images
into, only to be specified if `include_top` is True, and
if no `weights` argument is specified. 1000 is how many
ImageNet classes there are. Defaults to `1000`.
</td>
</tr><tr>
<td>
`classifier_activation`<a id="classifier_activation"></a>
</td>
<td>
A string or callable. The activation function to use
on the `"top"` layer. Ignored unless `include_top=True`. Set
`classifier_activation=None` to return the logits of the "top" layer.
When loading pretrained weights, `classifier_activation` can only
be `None` or `"softmax"`.
Defaults to `"softmax"`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../../tf/keras/Model.md"><code>keras.Model</code></a> instance.
</td>
</tr>

</table>

