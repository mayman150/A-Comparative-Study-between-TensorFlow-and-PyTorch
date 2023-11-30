description: Retrieves a live reference to the global dictionary of custom objects.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving.get_custom_objects" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.saving.get_custom_objects

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/object_registration.py#L75-L95">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Retrieves a live reference to the global dictionary of custom objects.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.utils.get_custom_objects`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.saving.get_custom_objects()
</code></pre>



<!-- Placeholder for "Used in" -->

Custom objects set using using `custom_object_scope` are not added to the
global dictionary of custom objects, and will not appear in the returned
dictionary.

#### Example:



```python
get_custom_objects().clear()
get_custom_objects()['MyObject'] = MyObject
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Global dictionary mapping registered class names to classes.
</td>
</tr>

</table>

