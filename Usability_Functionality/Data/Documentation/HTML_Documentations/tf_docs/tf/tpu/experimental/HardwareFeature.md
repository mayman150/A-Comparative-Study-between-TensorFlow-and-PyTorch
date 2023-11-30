description: class holds all the feature info about the TPU.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.HardwareFeature" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="EmbeddingFeature"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf.tpu.experimental.HardwareFeature

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_hardware_feature.py">View source</a>



class holds all the feature info about the TPU.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.HardwareFeature`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.HardwareFeature(
    tpu_hardware_feature_proto
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tpu_hardware_feature_proto`<a id="tpu_hardware_feature_proto"></a>
</td>
<td>
protobuf which describe the tpu hardware
feature.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`embedding_feature`<a id="embedding_feature"></a>
</td>
<td>
TPU embedding feature.
</td>
</tr><tr>
<td>
`num_embedding_devices_per_chip`<a id="num_embedding_devices_per_chip"></a>
</td>
<td>
Number of embedding accelerator devices per chip.
</td>
</tr>
</table>



## Child Classes
[`class EmbeddingFeature`](../../../tf/tpu/experimental/HardwareFeature/EmbeddingFeature.md)

