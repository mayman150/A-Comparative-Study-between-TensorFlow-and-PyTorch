description: Loads a model saved via model.save().

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving.load_model" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.saving.load_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/saving_api.py#L176-L264">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads a model saved via `model.save()`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.models.load_model`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.saving.load_model(
    filepath, custom_objects=None, compile=True, safe_mode=True, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filepath`<a id="filepath"></a>
</td>
<td>
`str` or `pathlib.Path` object, path to the saved model file.
</td>
</tr><tr>
<td>
`custom_objects`<a id="custom_objects"></a>
</td>
<td>
Optional dictionary mapping names
(strings) to custom classes or functions to be
considered during deserialization.
</td>
</tr><tr>
<td>
`compile`<a id="compile"></a>
</td>
<td>
Boolean, whether to compile the model after loading.
</td>
</tr><tr>
<td>
`safe_mode`<a id="safe_mode"></a>
</td>
<td>
Boolean, whether to disallow unsafe `lambda` deserialization.
When `safe_mode=False`, loading an object has the potential to
trigger arbitrary code execution. This argument is only
applicable to the Keras v3 model format. Defaults to True.
</td>
</tr>
</table>


SavedModel format arguments:
    options: Only applies to SavedModel format.
        Optional <a href="../../../tf/saved_model/LoadOptions.md"><code>tf.saved_model.LoadOptions</code></a> object that specifies
        SavedModel loading options.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Keras model instance. If the original model was compiled,
and the argument `compile=True` is set, then the returned model
will be compiled. Otherwise, the model will be left uncompiled.
</td>
</tr>

</table>



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

Note that the model variables may have different name values
(`var.name` property, e.g. `"dense_1/kernel:0"`) after being reloaded.
It is recommended that you use layer attributes to
access specific variables, e.g. `model.get_layer("dense_1").kernel`.