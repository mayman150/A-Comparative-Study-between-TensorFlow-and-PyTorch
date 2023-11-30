description: Saves a model as a TensorFlow SavedModel or HDF5 file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving.save_model" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.saving.save_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/saving_api.py#L49-L173">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Saves a model as a TensorFlow SavedModel or HDF5 file.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.models.save_model`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.saving.save_model(
    model, filepath, overwrite=True, save_format=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

See the [Serialization and Saving guide](
    https://keras.io/guides/serialization_and_saving/) for details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`<a id="model"></a>
</td>
<td>
Keras model instance to be saved.
</td>
</tr><tr>
<td>
`filepath`<a id="filepath"></a>
</td>
<td>
`str` or `pathlib.Path` object. Path where to save the model.
</td>
</tr><tr>
<td>
`overwrite`<a id="overwrite"></a>
</td>
<td>
Whether we should overwrite any existing model at the target
location, or instead ask the user via an interactive prompt.
</td>
</tr><tr>
<td>
`save_format`<a id="save_format"></a>
</td>
<td>
Either `"keras"`, `"tf"`, `"h5"`,
indicating whether to save the model
in the native Keras format (`.keras`),
in the TensorFlow SavedModel format (referred to as "SavedModel"
below), or in the legacy HDF5 format (`.h5`).
Defaults to `"tf"` in TF 2.X, and `"h5"` in TF 1.X.
</td>
</tr>
</table>


SavedModel format arguments:
    include_optimizer: Only applied to SavedModel and legacy HDF5 formats.
        If False, do not save the optimizer state. Defaults to True.
    signatures: Only applies to SavedModel format. Signatures to save
        with the SavedModel. See the `signatures` argument in
        <a href="../../../tf/saved_model/save.md"><code>tf.saved_model.save</code></a> for details.
    options: Only applies to SavedModel format.
        <a href="../../../tf/saved_model/SaveOptions.md"><code>tf.saved_model.SaveOptions</code></a> object that specifies SavedModel
        saving options.
    save_traces: Only applies to SavedModel format. When enabled, the
        SavedModel will store the function traces for each layer. This
        can be disabled, so that only the configs of each layer are stored.
        Defaults to `True`. Disabling this will decrease serialization time
        and reduce file size, but it requires that all custom layers/models
        implement a `get_config()` method.

#### Example:



```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
model.save("model.keras")
loaded_model = tf.keras.saving.load_model("model.keras")
x = tf.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))
```

Note that `model.save()` is an alias for <a href="../../../tf/keras/saving/save_model.md"><code>tf.keras.saving.save_model()</code></a>.

The SavedModel or HDF5 file contains:

- The model's configuration (architecture)
- The model's weights
- The model's optimizer's state (if any)

Thus models can be reinstantiated in the exact same state, without any of
the code used for model definition or training.

Note that the model weights may have different scoped names after being
loaded. Scoped names include the model/layer names, such as
`"dense_1/kernel:0"`. It is recommended that you use the layer properties to
access specific variables, e.g. `model.get_layer("dense_1").kernel`.

__SavedModel serialization format__

With `save_format="tf"`, the model and all trackable objects attached
to the it (e.g. layers and variables) are saved as a TensorFlow SavedModel.
The model config, weights, and optimizer are included in the SavedModel.
Additionally, for every Keras layer attached to the model, the SavedModel
stores:

* The config and metadata -- e.g. name, dtype, trainable status
* Traced call and loss functions, which are stored as TensorFlow
  subgraphs.

The traced functions allow the SavedModel format to save and load custom
layers without the original class definition.

You can choose to not save the traced functions by disabling the
`save_traces` option. This will decrease the time it takes to save the model
and the amount of disk space occupied by the output SavedModel. If you
enable this option, then you _must_ provide all custom class definitions
when loading the model. See the `custom_objects` argument in
<a href="../../../tf/keras/saving/load_model.md"><code>tf.keras.saving.load_model</code></a>.