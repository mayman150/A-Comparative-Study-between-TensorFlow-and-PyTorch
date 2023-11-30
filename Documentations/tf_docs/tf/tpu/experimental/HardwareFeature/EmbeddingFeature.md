description: Embedding feature flag strings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.HardwareFeature.EmbeddingFeature" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="UNSUPPORTED"/>
<meta itemprop="property" content="V1"/>
<meta itemprop="property" content="V2"/>
</div>

# tf.tpu.experimental.HardwareFeature.EmbeddingFeature

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_hardware_feature.py">View source</a>



Embedding feature flag strings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.HardwareFeature.EmbeddingFeature`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

UNSUPPORTED: No embedding lookup accelerator available on the tpu.
V1: Embedding lookup accelerator V1. The embedding lookup operation can only
    be placed at the beginning of computation. Only one instance of
    embedding
    lookup layer is allowed.
V2: Embedding lookup accelerator V2. The embedding lookup operation can be
    placed anywhere of the computation. Multiple instances of embedding
    lookup layer is allowed.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
UNSUPPORTED<a id="UNSUPPORTED"></a>
</td>
<td>
`<EmbeddingFeature.UNSUPPORTED: 'UNSUPPORTED'>`
</td>
</tr><tr>
<td>
V1<a id="V1"></a>
</td>
<td>
`<EmbeddingFeature.V1: 'V1'>`
</td>
</tr><tr>
<td>
V2<a id="V2"></a>
</td>
<td>
`<EmbeddingFeature.V2: 'V2'>`
</td>
</tr>
</table>

