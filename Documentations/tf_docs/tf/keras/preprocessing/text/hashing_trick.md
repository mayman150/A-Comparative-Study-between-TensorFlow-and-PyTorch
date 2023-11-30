description: Converts a text to a sequence of indexes in a fixed-size hashing space.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.text.hashing_trick" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.text.hashing_trick

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/text.py#L135-L191">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a text to a sequence of indexes in a fixed-size hashing space.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.text.hashing_trick(
    text,
    n,
    hash_function=None,
    filters=&#x27;!&quot;#$%&amp;()*+,-./:;&lt;=&gt;?@[\\]^_`{|}~\t\n&#x27;,
    lower=True,
    split=&#x27; &#x27;,
    analyzer=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: `tf.keras.text.preprocessing.hashing_trick` does not operate on
tensors and is not recommended for new code. Prefer
<a href="../../../../tf/keras/layers/Hashing.md"><code>tf.keras.layers.Hashing</code></a> which provides equivalent functionality through a
layer which accepts <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a> input. See the [preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers) for an
overview of preprocessing layers.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`text`<a id="text"></a>
</td>
<td>
Input text (string).
</td>
</tr><tr>
<td>
`n`<a id="n"></a>
</td>
<td>
Dimension of the hashing space.
</td>
</tr><tr>
<td>
`hash_function`<a id="hash_function"></a>
</td>
<td>
When `None` uses a python `hash` function. Can be 'md5'
or any function that takes in input a string and returns a int.
Note that 'hash' is not a stable hashing function, so
it is not consistent across different runs, while 'md5'
is a stable hashing function. Defaults to `None`.
</td>
</tr><tr>
<td>
`filters`<a id="filters"></a>
</td>
<td>
list (or concatenation) of characters to filter out, such as
punctuation. Default: ``!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n``,
includes basic punctuation, tabs, and newlines.
</td>
</tr><tr>
<td>
`lower`<a id="lower"></a>
</td>
<td>
boolean. Whether to set the text to lowercase.
</td>
</tr><tr>
<td>
`split`<a id="split"></a>
</td>
<td>
str. Separator for word splitting.
</td>
</tr><tr>
<td>
`analyzer`<a id="analyzer"></a>
</td>
<td>
function. Custom analyzer to split the text
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of integer word indices (unicity non-guaranteed).
`0` is a reserved index that won't be assigned to any word.
Two or more words may be assigned to the same index, due to possible
collisions by the hashing function.
The [probability](
    https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
of a collision is in relation to the dimension of the hashing space and
the number of distinct objects.
</td>
</tr>

</table>

