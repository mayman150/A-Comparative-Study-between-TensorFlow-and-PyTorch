description: Regularizer that encourages input vectors to be orthogonal to each other.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.regularizers.OrthogonalRegularizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.regularizers.OrthogonalRegularizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/regularizers.py#L328-L392">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Regularizer that encourages input vectors to be orthogonal to each other.

Inherits From: [`Regularizer`](../../../tf/keras/regularizers/Regularizer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.regularizers.orthogonal_regularizer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.regularizers.OrthogonalRegularizer(
    factor=0.01, mode=&#x27;rows&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

It can be applied to either the rows of a matrix (`mode="rows"`) or its
columns (`mode="columns"`). When applied to a `Dense` kernel of shape
`(input_dim, units)`, rows mode will seek to make the feature vectors
(i.e. the basis of the output space) orthogonal to each other.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`factor`<a id="factor"></a>
</td>
<td>
Float. The regularization factor. The regularization penalty will
be proportional to `factor` times the mean of the dot products between
the L2-normalized rows (if `mode="rows"`, or columns if
`mode="columns"`) of the inputs, excluding the product of each
row/column with itself.  Defaults to 0.01.
</td>
</tr><tr>
<td>
`mode`<a id="mode"></a>
</td>
<td>
String, one of `{"rows", "columns"}`. Defaults to `"rows"`. In rows
mode, the regularization effect seeks to make the rows of the input
orthogonal to each other. In columns mode, it seeks to make the columns
of the input orthogonal to each other.
</td>
</tr>
</table>



#### Example:



```
>>> regularizer = tf.keras.regularizers.OrthogonalRegularizer(factor=0.01)
>>> layer = tf.keras.layers.Dense(units=4, kernel_regularizer=regularizer)
```

## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/regularizers.py#L172-L190">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Creates a regularizer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same regularizer from the config
dictionary.

This method is used by Keras `model_to_estimator`, saving and
loading models to HDF5 formats, Keras model cloning, some visualization
utilities, and exporting models to and from JSON.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, typically the output of get_config.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A regularizer instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/regularizers.py#L391-L392">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config of the regularizer.

An regularizer config is a Python dictionary (serializable)
containing all configuration parameters of the regularizer.
The same regularizer can be reinstantiated later
(without any saved state) from this configuration.

This method is optional if you are just training and executing models,
exporting to and from SavedModels, or using weight checkpoints.

This method is required for Keras `model_to_estimator`, saving and
loading models to HDF5 formats, Keras model cloning, some visualization
utilities, and exporting models to and from JSON.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/regularizers.py#L368-L389">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    inputs
)
</code></pre>

Compute a regularization penalty from an input tensor.




