description: Serializes a LearningRateSchedule into a JSON-compatible dict.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.schedules.serialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.optimizers.schedules.serialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/schedules/learning_rate_schedule.py#L1191-L1215">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Serializes a `LearningRateSchedule` into a JSON-compatible dict.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.schedules.serialize(
    learning_rate_schedule, use_legacy_format=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate_schedule`<a id="learning_rate_schedule"></a>
</td>
<td>
The `LearningRateSchedule` object to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable dict representing the object's config.
</td>
</tr>

</table>



#### Example:



```
>>> lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
...   0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
>>> tf.keras.optimizers.schedules.serialize(lr_schedule)
{'module': 'keras.optimizers.schedules',
'class_name': 'ExponentialDecay', 'config': {...},
'registered_name': None}
```