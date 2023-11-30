description: Settings for simulated quantization of the tpu embedding table.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.embedding.QuantizationConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.tpu.experimental.embedding.QuantizationConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_v2_utils.py">View source</a>



Settings for simulated quantization of the tpu embedding table.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.embedding.QuantizationConfig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.embedding.QuantizationConfig(
    num_buckets: int, lower: float, upper: float
)
</code></pre>



<!-- Placeholder for "Used in" -->

When simulated quantization is enabled, the results of the embedding lookup
are clipped and quantized according to the settings here before the combiner
is applied.

For example, to quantize `input` the following is done:
```python
if input < lower
  input = lower
if input > upper
  input = upper
quantum = (upper - lower) / (num_buckets - 1)
input = math.floor((input - lower) / quantum + 0.5) * quantium + lower
```

See tensorflow/core/protobuf/tpu/optimization_parameters.proto for more
details.

NOTE: This does not change the storage type of the embedding table, that will
continue to be float32 as will the saved variable in the checkpoint. You will
have to manually quantize the variable (typically with the same algorithm and
settings as above) manually.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_buckets`<a id="num_buckets"></a>
</td>
<td>
The number of quantization buckets, must be atleast 2.
</td>
</tr><tr>
<td>
`lower`<a id="lower"></a>
</td>
<td>
The lower bound for the quantization range.
</td>
</tr><tr>
<td>
`upper`<a id="upper"></a>
</td>
<td>
The upper bound for the quantization range.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if `num_buckets` is less than 2.
</td>
</tr>
</table>



