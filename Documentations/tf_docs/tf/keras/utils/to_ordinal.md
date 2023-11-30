description: Converts a class vector (integers) to an ordinal regression matrix.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.to_ordinal" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.to_ordinal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/np_utils.py#L80-L125">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a class vector (integers) to an ordinal regression matrix.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.to_ordinal(
    y, num_classes=None, dtype=&#x27;float32&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This utility encodes class vector to ordinal regression/classification
matrix where each sample is indicated by a row and rank of that sample is
indicated by number of ones in that row.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`y`<a id="y"></a>
</td>
<td>
Array-like with class values to be converted into a matrix
(integers from 0 to `num_classes - 1`).
</td>
</tr><tr>
<td>
`num_classes`<a id="num_classes"></a>
</td>
<td>
Total number of classes. If `None`, this would be inferred
as `max(y) + 1`.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The data type expected by the input. Default: `'float32'`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An ordinal regression matrix representation of the input as a NumPy
array. The class axis is placed last.
</td>
</tr>

</table>



#### Example:



```
>>> a = tf.keras.utils.to_ordinal([0, 1, 2, 3], num_classes=4)
>>> print(a)
[[0. 0. 0.]
 [1. 0. 0.]
 [1. 1. 0.]
 [1. 1. 1.]]
```