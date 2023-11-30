description: ExportArchive is used to write SavedModel artifacts (e.g. for inference).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.export.ExportArchive" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_endpoint"/>
<meta itemprop="property" content="add_variable_collection"/>
<meta itemprop="property" content="track"/>
<meta itemprop="property" content="write_out"/>
</div>

# tf.keras.export.ExportArchive

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/export/export_lib.py#L26-L393">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



ExportArchive is used to write SavedModel artifacts (e.g. for inference).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.export.ExportArchive()
</code></pre>



<!-- Placeholder for "Used in" -->

If you have a Keras model or layer that you want to export as SavedModel for
serving (e.g. via TensorFlow-Serving), you can use `ExportArchive`
to configure the different serving endpoints you need to make available,
as well as their signatures. Simply instantiate an `ExportArchive`,
use `track()` to register the layer(s) or model(s) to be used,
then use the `add_endpoint()` method to register a new serving endpoint.
When done, use the `write_out()` method to save the artifact.

The resulting artifact is a SavedModel and can be reloaded via
<a href="../../../tf/saved_model/load.md"><code>tf.saved_model.load</code></a>.

#### Examples:



Here's how to export a model for inference.

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
export_archive.write_out("path/to/location")

# Elsewhere, we can reload the artifact and serve it.
# The endpoint we added is available as a method:
serving_model = tf.saved_model.load("path/to/location")
outputs = serving_model.serve(inputs)
```

Here's how to export a model with one endpoint for inference and one
endpoint for a training-mode forward pass (e.g. with dropout on).

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="call_inference",
    fn=lambda x: model.call(x, training=False),
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
export_archive.add_endpoint(
    name="call_training",
    fn=lambda x: model.call(x, training=True),
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
export_archive.write_out("path/to/location")
```

**Note on resource tracking:**

`ExportArchive` is able to automatically track all `tf.Variables` used
by its endpoints, so most of the time calling `.track(model)`
is not strictly required. However, if your model uses lookup layers such
as `IntegerLookup`, `StringLookup`, or `TextVectorization`,
it will need to be tracked explicitly via `.track(model)`.

Explicit tracking is also required if you need to be able to access
the properties `variables`, `trainable_variables`, or
`non_trainable_variables` on the revived archive.

## Methods

<h3 id="add_endpoint"><code>add_endpoint</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/export/export_lib.py#L132-L276">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_endpoint(
    name, fn, input_signature=None
)
</code></pre>

Register a new serving endpoint.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`name`
</td>
<td>
Str, name of the endpoint.
</td>
</tr><tr>
<td>
`fn`
</td>
<td>
A function. It should only leverage resources
(e.g. <a href="../../../tf/Variable.md"><code>tf.Variable</code></a> objects or <a href="../../../tf/lookup/StaticHashTable.md"><code>tf.lookup.StaticHashTable</code></a>
objects) that are available on the models/layers
tracked by the `ExportArchive` (you can call `.track(model)`
to track a new model).
The shape and dtype of the inputs to the function must be
known. For that purpose, you can either 1) make sure that
`fn` is a <a href="../../../tf/function.md"><code>tf.function</code></a> that has been called at least once, or
2) provide an `input_signature` argument that specifies the
shape and dtype of the inputs (see below).
</td>
</tr><tr>
<td>
`input_signature`
</td>
<td>
Used to specify the shape and dtype of the
inputs to `fn`. List of <a href="../../../tf/TensorSpec.md"><code>tf.TensorSpec</code></a> objects (one
per positional input argument of `fn`). Nested arguments are
allowed (see below for an example showing a Functional model
with 2 input arguments).
</td>
</tr>
</table>



#### Example:



Adding an endpoint using the `input_signature` argument when the
model has a single input argument:

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
```

Adding an endpoint using the `input_signature` argument when the
model has two positional input arguments:

```python
export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[
        tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    ],
)
```

Adding an endpoint using the `input_signature` argument when the
model has one input argument that is a list of 2 tensors (e.g.
a Functional model with 2 inputs):

```python
model = keras.Model(inputs=[x1, x2], outputs=outputs)

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[
        [
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        ],
    ],
)
```

This also works with dictionary inputs:

```python
model = keras.Model(inputs={"x1": x1, "x2": x2}, outputs=outputs)

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[
        {
            "x1": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            "x2": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        },
    ],
)
```

Adding an endpoint that is a <a href="../../../tf/function.md"><code>tf.function</code></a>:

```python
@tf.function()
def serving_fn(x):
    return model(x)

# The function must be traced, i.e. it must be called at least once.
serving_fn(tf.random.normal(shape=(2, 3)))

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(name="serve", fn=serving_fn)
```

<h3 id="add_variable_collection"><code>add_variable_collection</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/export/export_lib.py#L278-L318">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_variable_collection(
    name, variables
)
</code></pre>

Register a set of variables to be retrieved after reloading.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`name`
</td>
<td>
The string name for the collection.
</td>
</tr><tr>
<td>
`variables`
</td>
<td>
A tuple/list/set of <a href="../../../tf/Variable.md"><code>tf.Variable</code></a> instances.
</td>
</tr>
</table>



#### Example:



```python
export_archive = ExportArchive()
export_archive.track(model)
# Register an endpoint
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
)
# Save a variable collection
export_archive.add_variable_collection(
    name="optimizer_variables", variables=model.optimizer.variables)
export_archive.write_out("path/to/location")

# Reload the object
revived_object = tf.saved_model.load("path/to/location")
# Retrieve the variables
optimizer_variables = revived_object.optimizer_variables
```

<h3 id="track"><code>track</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/export/export_lib.py#L101-L130">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>track(
    resource
)
</code></pre>

Track the variables (and other assets) of a layer or model.


<h3 id="write_out"><code>write_out</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/export/export_lib.py#L320-L362">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write_out(
    filepath, options=None
)
</code></pre>

Write the corresponding SavedModel to disk.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`filepath`
</td>
<td>
`str` or `pathlib.Path` object.
Path where to save the artifact.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
<a href="../../../tf/saved_model/SaveOptions.md"><code>tf.saved_model.SaveOptions</code></a> object that specifies
SavedModel saving options.
</td>
</tr>
</table>


**Note on TF-Serving**: all endpoints registered via `add_endpoint()`
are made visible for TF-Serving in the SavedModel artifact. In addition,
the first endpoint registered is made visible under the alias
`"serving_default"` (unless an endpoint with the name
`"serving_default"` was already registered manually),
since TF-Serving requires this endpoint to be set.



