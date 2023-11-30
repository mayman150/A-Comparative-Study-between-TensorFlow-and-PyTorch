description: Registers an object with the Keras serialization framework.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving.register_keras_serializable" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.saving.register_keras_serializable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/object_registration.py#L98-L158">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Registers an object with the Keras serialization framework.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.utils.register_keras_serializable`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.saving.register_keras_serializable(
    package=&#x27;Custom&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This decorator injects the decorated class or function into the Keras custom
object dictionary, so that it can be serialized and deserialized without
needing an entry in the user-provided custom object dict. It also injects a
function that Keras will call to get the object's serializable string key.

Note that to be serialized and deserialized, classes must implement the
`get_config()` method. Functions do not have this requirement.

The object will be registered under the key 'package>name' where `name`,
defaults to the object name if not passed.

#### Example:



```python
# Note that `'my_package'` is used as the `package` argument here, and since
# the `name` argument is not provided, `'MyDense'` is used as the `name`.
@keras.saving.register_keras_serializable('my_package')
class MyDense(keras.layers.Dense):
  pass

assert keras.saving.get_registered_object('my_package>MyDense') == MyDense
assert keras.saving.get_registered_name(MyDense) == 'my_package>MyDense'
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`package`<a id="package"></a>
</td>
<td>
The package that this class belongs to. This is used for the
`key` (which is `"package>name"`) to idenfify the class. Note that this
is the first argument passed into the decorator.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name to serialize this class under in this package. If not
provided or `None`, the class' name will be used (note that this is the
case when the decorator is used with only one argument, which becomes
the `package`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator that registers the decorated class with the passed names.
</td>
</tr>

</table>

