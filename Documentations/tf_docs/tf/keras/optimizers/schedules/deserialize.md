description: Instantiates a LearningRateSchedule object from a serialized form.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.schedules.deserialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.optimizers.schedules.deserialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/schedules/learning_rate_schedule.py#L1218-L1259">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Instantiates a `LearningRateSchedule` object from a serialized form.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.schedules.deserialize(
    config, custom_objects=None, use_legacy_format=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`config`<a id="config"></a>
</td>
<td>
The serialized form of the `LearningRateSchedule`.
Dictionary of the form {'class_name': str, 'config': dict}.
</td>
</tr><tr>
<td>
`custom_objects`<a id="custom_objects"></a>
</td>
<td>
A dictionary mapping class names (or function names) of
custom (non-Keras) objects to class/functions.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `LearningRateSchedule` object.
</td>
</tr>

</table>



#### Example:



```python
# Configuration for PolynomialDecay
config = {
  'class_name': 'PolynomialDecay',
  'config': {'cycle': False,
    'decay_steps': 10000,
    'end_learning_rate': 0.01,
    'initial_learning_rate': 0.1,
    'name': None,
    'power': 0.5}}
lr_schedule = tf.keras.optimizers.schedules.deserialize(config)
```