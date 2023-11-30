description: Quantizes the SavedModel with the given quantization options.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.quantization.experimental.quantize_saved_model" />
<meta itemprop="path" content="Stable" />
</div>

# tf.quantization.experimental.quantize_saved_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.py">View source</a>



Quantizes the SavedModel with the given quantization options.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.quantization.experimental.quantize_saved_model`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.quantization.experimental.quantize_saved_model(
    saved_model_path: str,
    output_directory: Optional[str] = None,
    quantization_options: Optional[<a href="../../../tf/quantization/experimental/QuantizationOptions.md"><code>tf.quantization.experimental.QuantizationOptions</code></a>] = None,
    representative_dataset: Optional[repr_dataset.RepresentativeDatasetOrMapping] = None,
    *,
    overwrite_output_directory: bool = False
) -> autotrackable.AutoTrackable
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example usage:


```python
# Quantizing a model trained with QAT.
quantization_options = tf.quantization.experimental.QuantizationOptions(
    signature_keys=['your_signature_key'],
)
tf.quantization.experimental.quantize_saved_model(
    '/tmp/input_model',
    '/tmp/output_model',
    quantization_options=quantization_options,
)

# When quantizing a model trained without QAT (Post-Training Quantization),
# a representative dataset is required.
representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                      for _ in range(256)]
tf.quantization.experimental.quantize_saved_model(
    '/tmp/input_model',
    '/tmp/output_model',
    quantization_options=quantization_options,
    representative_dataset={'your_signature_key': representative_dataset},
  )

# In addition to preset quantization methods, fine-grained control of
# quantization for each component is also supported.
_QuantizationComponentSpec = (
    tf.quantization.experimental.QuantizationComponentSpec
)
quantization_options = tf.quantization.experimental.QuantizationOptions(
    signature_keys=['your_signature_key'],
    quantization_method=tf.quantization.experimental.QuantizationMethod(
        quantization_component_specs=[
            _QuantizationComponentSpec(
                quantization_component=(
                    _QuantizationComponentSpec.COMPONENT_ACTIVATION
                ),
                tensor_type=_QuantizationComponentSpec.TENSORTYPE_INT_8,
            )
        ]
    )
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
`saved_model_path`<a id="saved_model_path"></a>
</td>
<td>
Path to the saved model. When representative_dataset is
not provided, this should be a model trained with QAT.
</td>
</tr><tr>
<td>
`output_directory`<a id="output_directory"></a>
</td>
<td>
The path to save the output SavedModel. Set
`overwrite_output_directory` to `True` to overwrite any existing contents
in the directory if not empty.
</td>
</tr><tr>
<td>
`quantization_options`<a id="quantization_options"></a>
</td>
<td>
A set of options for quantization. If None, it uses
post-training static range quantization with XLA opset by default.
</td>
</tr><tr>
<td>
`representative_dataset`<a id="representative_dataset"></a>
</td>
<td>
an iterator that returns a dictionary of {input_key:
input_value} or a map from signature key to a dictionary of {input_key:
input_value} that feeds calibration data for quantizing model. The
representative should be provided when the model is a PTQ model. It can be
provided either via this parameter or via the `representative_datasets`
field in `QuantizationOptions`.
</td>
</tr><tr>
<td>
`overwrite_output_directory`<a id="overwrite_output_directory"></a>
</td>
<td>
If set to true, overwrites the output directory
iff it isn't empty. The default value is false.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A SavedModel object with TF quantization applied, or None if no quantization
is performed.
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
When 1) representative_dataset is not provided for non QAT model
for enabling static range quantization, 2) invalid value is provided as
a quantization method, or 3) provide representative dataset via both
argument and QuantizationOptions.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
When the specified quantization method is not yet supported.
</td>
</tr>
</table>

