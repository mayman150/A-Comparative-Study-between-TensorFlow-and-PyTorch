description: Retrieve the object by deserializing the config dict.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving.deserialize_keras_object" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.saving.deserialize_keras_object

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/serialization_lib.py#L406-L740">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Retrieve the object by deserializing the config dict.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.utils.deserialize_keras_object`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.saving.deserialize_keras_object(
    config, custom_objects=None, safe_mode=True, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The config dict is a Python dictionary that consists of a set of key-value
pairs, and represents a Keras object, such as an `Optimizer`, `Layer`,
`Metrics`, etc. The saving and loading library uses the following keys to
record information of a Keras object:

- `class_name`: String. This is the name of the class,
  as exactly defined in the source
  code, such as "LossesContainer".
- `config`: Dict. Library-defined or user-defined key-value pairs that store
  the configuration of the object, as obtained by `object.get_config()`.
- `module`: String. The path of the python module, such as
  "keras.engine.compile_utils". Built-in Keras classes
  expect to have prefix `keras`.
- `registered_name`: String. The key the class is registered under via
  <a href="../../../tf/keras/saving/register_keras_serializable.md"><code>keras.saving.register_keras_serializable(package, name)</code></a> API. The key has
  the format of '{package}>{name}', where `package` and `name` are the
  arguments passed to `register_keras_serializable()`. If `name` is not
  provided, it uses the class name. If `registered_name` successfully
  resolves to a class (that was registered), the `class_name` and `config`
  values in the dict will not be used. `registered_name` is only used for
  non-built-in classes.

For example, the following dictionary represents the built-in Adam optimizer
with the relevant config:

```python
dict_structure = {
    "class_name": "Adam",
    "config": {
        "amsgrad": false,
        "beta_1": 0.8999999761581421,
        "beta_2": 0.9990000128746033,
        "decay": 0.0,
        "epsilon": 1e-07,
        "learning_rate": 0.0010000000474974513,
        "name": "Adam"
    },
    "module": "keras.optimizers",
    "registered_name": None
}
# Returns an `Adam` instance identical to the original one.
deserialize_keras_object(dict_structure)
```

If the class does not have an exported Keras namespace, the library tracks
it by its `module` and `class_name`. For example:

```python
dict_structure = {
  "class_name": "LossesContainer",
  "config": {
      "losses": [...],
      "total_loss_mean": {...},
  },
  "module": "keras.engine.compile_utils",
  "registered_name": "LossesContainer"
}

# Returns a `LossesContainer` instance identical to the original one.
deserialize_keras_object(dict_structure)
```

And the following dictionary represents a user-customized `MeanSquaredError`
loss:

```python
@keras.saving.register_keras_serializable(package='my_package')
class ModifiedMeanSquaredError(keras.losses.MeanSquaredError):
  ...

dict_structure = {
    "class_name": "ModifiedMeanSquaredError",
    "config": {
        "fn": "mean_squared_error",
        "name": "mean_squared_error",
        "reduction": "auto"
    },
    "registered_name": "my_package>ModifiedMeanSquaredError"
}
# Returns the `ModifiedMeanSquaredError` object
deserialize_keras_object(dict_structure)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`config`<a id="config"></a>
</td>
<td>
Python dict describing the object.
</td>
</tr><tr>
<td>
`custom_objects`<a id="custom_objects"></a>
</td>
<td>
Python dict containing a mapping between custom
object names the corresponding classes or functions.
</td>
</tr><tr>
<td>
`safe_mode`<a id="safe_mode"></a>
</td>
<td>
Boolean, whether to disallow unsafe `lambda` deserialization.
When `safe_mode=False`, loading an object has the potential to
trigger arbitrary code execution. This argument is only
applicable to the Keras v3 model format. Defaults to `True`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The object described by the `config` dictionary.
</td>
</tr>

</table>

