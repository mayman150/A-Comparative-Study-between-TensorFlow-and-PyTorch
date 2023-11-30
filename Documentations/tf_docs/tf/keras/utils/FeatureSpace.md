description: One-stop utility for preprocessing and encoding structured data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.FeatureSpace" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="cross"/>
<meta itemprop="property" content="feature"/>
<meta itemprop="property" content="float"/>
<meta itemprop="property" content="float_discretized"/>
<meta itemprop="property" content="float_normalized"/>
<meta itemprop="property" content="float_rescaled"/>
<meta itemprop="property" content="get_encoded_features"/>
<meta itemprop="property" content="get_inputs"/>
<meta itemprop="property" content="integer_categorical"/>
<meta itemprop="property" content="integer_hashed"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="string_categorical"/>
<meta itemprop="property" content="string_hashed"/>
</div>

# tf.keras.utils.FeatureSpace

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L89-L772">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



One-stop utility for preprocessing and encoding structured data.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.FeatureSpace(
    features,
    output_mode=&#x27;concat&#x27;,
    crosses=None,
    crossing_dim=32,
    hashing_dim=32,
    num_discretization_bins=32
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`feature_names`<a id="feature_names"></a>
</td>
<td>
Dict mapping the names of your features to their
type specification, e.g. `{"my_feature": "integer_categorical"}`
or `{"my_feature": FeatureSpace.integer_categorical()}`.
For a complete list of all supported types, see
"Available feature types" paragraph below.
</td>
</tr><tr>
<td>
`output_mode`<a id="output_mode"></a>
</td>
<td>
One of `"concat"` or `"dict"`. In concat mode, all
features get concatenated together into a single vector.
In dict mode, the FeatureSpace returns a dict of individually
encoded features (with the same keys as the input dict keys).
</td>
</tr><tr>
<td>
`crosses`<a id="crosses"></a>
</td>
<td>
List of features to be crossed together, e.g.
`crosses=[("feature_1", "feature_2")]`. The features will be
"crossed" by hashing their combined value into
a fixed-length vector.
</td>
</tr><tr>
<td>
`crossing_dim`<a id="crossing_dim"></a>
</td>
<td>
Default vector size for hashing crossed features.
Defaults to `32`.
</td>
</tr><tr>
<td>
`hashing_dim`<a id="hashing_dim"></a>
</td>
<td>
Default vector size for hashing features of type
`"integer_hashed"` and `"string_hashed"`. Defaults to `32`.
</td>
</tr><tr>
<td>
`num_discretization_bins`<a id="num_discretization_bins"></a>
</td>
<td>
Default number of bins to be used for
discretizing features of type `"float_discretized"`.
Defaults to `32`.
</td>
</tr>
</table>


**Available feature types:**

Note that all features can be referred to by their string name,
e.g. `"integer_categorical"`. When using the string name, the default
argument values are used.

```python
# Plain float values.
FeatureSpace.float(name=None)

# Float values to be preprocessed via featurewise standardization
# (i.e. via a `keras.layers.Normalization` layer).
FeatureSpace.float_normalized(name=None)

# Float values to be preprocessed via linear rescaling
# (i.e. via a `keras.layers.Rescaling` layer).
FeatureSpace.float_rescaled(scale=1., offset=0., name=None)

# Float values to be discretized. By default, the discrete
# representation will then be one-hot encoded.
FeatureSpace.float_discretized(
    num_bins, bin_boundaries=None, output_mode="one_hot", name=None)

# Integer values to be indexed. By default, the discrete
# representation will then be one-hot encoded.
FeatureSpace.integer_categorical(
    max_tokens=None, num_oov_indices=1, output_mode="one_hot", name=None)

# String values to be indexed. By default, the discrete
# representation will then be one-hot encoded.
FeatureSpace.string_categorical(
    max_tokens=None, num_oov_indices=1, output_mode="one_hot", name=None)

# Integer values to be hashed into a fixed number of bins.
# By default, the discrete representation will then be one-hot encoded.
FeatureSpace.integer_hashed(num_bins, output_mode="one_hot", name=None)

# String values to be hashed into a fixed number of bins.
# By default, the discrete representation will then be one-hot encoded.
FeatureSpace.string_hashed(num_bins, output_mode="one_hot", name=None)
```

#### Examples:



**Basic usage with a dict of input data:**

```python
raw_data = {
    "float_values": [0.0, 0.1, 0.2, 0.3],
    "string_values": ["zero", "one", "two", "three"],
    "int_values": [0, 1, 2, 3],
}
dataset = tf.data.Dataset.from_tensor_slices(raw_data)

feature_space = FeatureSpace(
    features={
        "float_values": "float_normalized",
        "string_values": "string_categorical",
        "int_values": "integer_categorical",
    },
    crosses=[("string_values", "int_values")],
    output_mode="concat",
)
# Before you start using the FeatureSpace,
# you must `adapt()` it on some data.
feature_space.adapt(dataset)

# You can call the FeatureSpace on a dict of data (batched or unbatched).
output_vector = feature_space(raw_data)
```

**Basic usage with <a href="../../../tf/data.md"><code>tf.data</code></a>:**

```python
# Unlabeled data
preprocessed_ds = unlabeled_dataset.map(feature_space)

# Labeled data
preprocessed_ds = labeled_dataset.map(lambda x, y: (feature_space(x), y))
```

**Basic usage with the Keras Functional API:**

```python
# Retrieve a dict Keras Input objects
inputs = feature_space.get_inputs()
# Retrieve the corresponding encoded Keras tensors
encoded_features = feature_space.get_encoded_features()
# Build a Functional model
outputs = keras.layers.Dense(1, activation="sigmoid")(encoded_features)
model = keras.Model(inputs, outputs)
```

**Customizing each feature or feature cross:**

```python
feature_space = FeatureSpace(
    features={
        "float_values": FeatureSpace.float_normalized(),
        "string_values": FeatureSpace.string_categorical(max_tokens=10),
        "int_values": FeatureSpace.integer_categorical(max_tokens=10),
    },
    crosses=[
        FeatureSpace.cross(("string_values", "int_values"), crossing_dim=32)
    ],
    output_mode="concat",
)
```

**Returning a dict of integer-encoded features:**

```python
feature_space = FeatureSpace(
    features={
        "string_values": FeatureSpace.string_categorical(output_mode="int"),
        "int_values": FeatureSpace.integer_categorical(output_mode="int"),
    },
    crosses=[
        FeatureSpace.cross(
            feature_names=("string_values", "int_values"),
            crossing_dim=32,
            output_mode="int",
        )
    ],
    output_mode="dict",
)
```

**Specifying your own Keras preprocessing layer:**

```python
# Let's say that one of the features is a short text paragraph that
# we want to encode as a vector (one vector per paragraph) via TF-IDF.
data = {
    "text": ["1st string", "2nd string", "3rd string"],
}

# There's a Keras layer for this: TextVectorization.
custom_layer = layers.TextVectorization(output_mode="tf_idf")

# We can use FeatureSpace.feature to create a custom feature
# that will use our preprocessing layer.
feature_space = FeatureSpace(
    features={
        "text": FeatureSpace.feature(
            preprocessor=custom_layer, dtype="string", output_mode="float"
        ),
    },
    output_mode="concat",
)
feature_space.adapt(tf.data.Dataset.from_tensor_slices(data))
output_vector = feature_space(data)
```

**Retrieving the underlying Keras preprocessing layers:**

```python
# The preprocessing layer of each feature is available in `.preprocessors`.
preprocessing_layer = feature_space.preprocessors["feature1"]

# The crossing layer of each feature cross is available in `.crossers`.
# It's an instance of keras.layers.HashedCrossing.
crossing_layer = feature_space.crossers["feature1_X_feature2"]
```

**Saving and reloading a FeatureSpace:**

```python
feature_space.save("myfeaturespace.keras")
reloaded_feature_space = keras.models.load_model("myfeaturespace.keras")
```

## Methods

<h3 id="adapt"><code>adapt</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L516-L550">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adapt(
    dataset
)
</code></pre>




<h3 id="cross"><code>cross</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L288-L290">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>cross(
    feature_names, crossing_dim, output_mode=&#x27;one_hot&#x27;
)
</code></pre>




<h3 id="feature"><code>feature</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L292-L294">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>feature(
    dtype, preprocessor, output_mode
)
</code></pre>




<h3 id="float"><code>float</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L296-L306">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>float(
    name=None
)
</code></pre>




<h3 id="float_discretized"><code>float_discretized</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L328-L340">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>float_discretized(
    num_bins, bin_boundaries=None, output_mode=&#x27;one_hot&#x27;, name=None
)
</code></pre>




<h3 id="float_normalized"><code>float_normalized</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L318-L326">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>float_normalized(
    name=None
)
</code></pre>




<h3 id="float_rescaled"><code>float_rescaled</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L308-L316">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>float_rescaled(
    scale=1.0, offset=0.0, name=None
)
</code></pre>




<h3 id="get_encoded_features"><code>get_encoded_features</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L556-L566">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_encoded_features()
</code></pre>




<h3 id="get_inputs"><code>get_inputs</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L552-L554">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_inputs()
</code></pre>




<h3 id="integer_categorical"><code>integer_categorical</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L342-L358">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>integer_categorical(
    max_tokens=None,
    num_oov_indices=1,
    output_mode=&#x27;one_hot&#x27;,
    name=None
)
</code></pre>




<h3 id="integer_hashed"><code>integer_hashed</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L388-L396">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>integer_hashed(
    num_bins, output_mode=&#x27;one_hot&#x27;, name=None
)
</code></pre>




<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L756-L766">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    filepath
)
</code></pre>

Save the `FeatureSpace` instance to a `.keras` file.

You can reload it via <a href="../../../tf/keras/saving/load_model.md"><code>keras.models.load_model()</code></a>:

```python
feature_space.save("myfeaturespace.keras")
reloaded_feature_space = keras.models.load_model("myfeaturespace.keras")
```

<h3 id="string_categorical"><code>string_categorical</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L360-L376">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>string_categorical(
    max_tokens=None,
    num_oov_indices=1,
    output_mode=&#x27;one_hot&#x27;,
    name=None
)
</code></pre>




<h3 id="string_hashed"><code>string_hashed</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/feature_space.py#L378-L386">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>string_hashed(
    num_bins, output_mode=&#x27;one_hot&#x27;, name=None
)
</code></pre>






