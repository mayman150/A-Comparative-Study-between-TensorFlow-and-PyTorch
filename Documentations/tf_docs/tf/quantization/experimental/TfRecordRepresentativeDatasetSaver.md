description: Representative dataset saver in TFRecord format.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.quantization.experimental.TfRecordRepresentativeDatasetSaver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="save"/>
</div>

# tf.quantization.experimental.TfRecordRepresentativeDatasetSaver

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/compiler/mlir/quantization/tensorflow/python/representative_dataset.py">View source</a>



Representative dataset saver in TFRecord format.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.quantization.experimental.TfRecordRepresentativeDatasetSaver`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.quantization.experimental.TfRecordRepresentativeDatasetSaver(
    path_map: Mapping[str, os.PathLike[str]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

Saves representative datasets for quantization calibration in TFRecord format.
The samples are serialized as `RepresentativeDataSample`.

The `save` method return a signature key to `RepresentativeDatasetFile` map,
which can be used for QuantizationOptions.

#### Example usage:



```python
# Creating the representative dataset.
representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                      for _ in range(256)]

# Saving to a TFRecord file.
dataset_file_map = (
  tf.quantization.experimental.TfRecordRepresentativeDatasetSaver(
        path_map={'serving_default': '/tmp/representative_dataset_path'}
    ).save({'serving_default': representative_dataset})
)

# Using in QuantizationOptions.
quantization_options = tf.quantization.experimental.QuantizationOptions(
    signature_keys=['serving_default'],
    representative_datasets=dataset_file_map,
)
tf.quantization.experimental.quantize_saved_model(
    '/tmp/input_model',
    '/tmp/output_model',
    quantization_options=quantization_options,
)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path_map`<a id="path_map"></a>
</td>
<td>
Signature def key -> path mapping. Each path is a TFRecord file
to which a `RepresentativeDataset` is saved. The signature def keys
should be a subset of the `SignatureDef` keys of the
`representative_dataset` argument of the `save()` call.
</td>
</tr>
</table>



## Methods

<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/compiler/mlir/quantization/tensorflow/python/representative_dataset.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    representative_dataset: RepresentativeDatasetMapping
) -> Mapping[str, _RepresentativeDatasetFile]
</code></pre>

Saves the representative dataset.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`representative_dataset`
</td>
<td>
Signature def key -> representative dataset
mapping. Each dataset is saved in a separate TFRecord file whose path
matches the signature def key of `path_map`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
When the signature def key in `representative_dataset` is not
present in the `path_map`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A map from signature key to the RepresentativeDatasetFile instance
contains the path to the saved file.
</td>
</tr>

</table>





