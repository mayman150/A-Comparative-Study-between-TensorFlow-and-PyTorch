description: One-hot encodes a text into a list of word indexes of size n.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.text.one_hot" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.text.one_hot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/text.py#L84-L132">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



One-hot encodes a text into a list of word indexes of size `n`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.text.one_hot(
    input_text,
    n,
    filters=&#x27;!&quot;#$%&amp;()*+,-./:;&lt;=&gt;?@[\\]^_`{|}~\t\n&#x27;,
    lower=True,
    split=&#x27; &#x27;,
    analyzer=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: `tf.keras.text.preprocessing.one_hot` does not operate on
tensors and is not recommended for new code. Prefer
<a href="../../../../tf/keras/layers/Hashing.md"><code>tf.keras.layers.Hashing</code></a> with `output_mode='one_hot'` which provides
equivalent functionality through a layer which accepts <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a> input.
See the [preprocessing layer guide]
(https://www.tensorflow.org/guide/keras/preprocessing_layers) for an
overview of preprocessing layers.

This function receives as input a string of text and returns a
list of encoded integers each corresponding to a word (or token)
in the given input string.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_text`<a id="input_text"></a>
</td>
<td>
Input text (string).
</td>
</tr><tr>
<td>
`n`<a id="n"></a>
</td>
<td>
int. Size of vocabulary.
</td>
</tr><tr>
<td>
`filters`<a id="filters"></a>
</td>
<td>
list (or concatenation) of characters to filter out, such as
punctuation. Default:
```
'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n
```,
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
List of integers in `[1, n]`. Each integer encodes a word
(unicity non-guaranteed).
</td>
</tr>

</table>

