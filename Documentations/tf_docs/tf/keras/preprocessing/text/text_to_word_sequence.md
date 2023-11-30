description: Converts a text to a sequence of words (or tokens).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.text.text_to_word_sequence" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.text.text_to_word_sequence

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/text.py#L39-L81">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts a text to a sequence of words (or tokens).


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.text.text_to_word_sequence(
    input_text,
    filters=&#x27;!&quot;#$%&amp;()*+,-./:;&lt;=&gt;?@[\\]^_`{|}~\t\n&#x27;,
    lower=True,
    split=&#x27; &#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/text/text_to_word_sequence.md"><code>tf.keras.preprocessing.text.text_to_word_sequence</code></a> does not
operate on tensors and is not recommended for new code. Prefer
<a href="../../../../tf/strings/regex_replace.md"><code>tf.strings.regex_replace</code></a> and <a href="../../../../tf/strings/split.md"><code>tf.strings.split</code></a> which provide equivalent
functionality and accept <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a> input. For an overview of text handling
in Tensorflow, see the [text loading tutorial]
(https://www.tensorflow.org/tutorials/load_data/text).

This function transforms a string of text into a list of words
while ignoring `filters` which include punctuations by default.

```
>>> sample_text = 'This is a sample sentence.'
>>> tf.keras.preprocessing.text.text_to_word_sequence(sample_text)
['this', 'is', 'a', 'sample', 'sentence']
```

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
`filters`<a id="filters"></a>
</td>
<td>
list (or concatenation) of characters to filter out, such as
punctuation. Default: ``'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\\t\\n'``,
  includes basic punctuation, tabs, and newlines.
</td>
</tr><tr>
<td>
`lower`<a id="lower"></a>
</td>
<td>
boolean. Whether to convert the input to lowercase.
</td>
</tr><tr>
<td>
`split`<a id="split"></a>
</td>
<td>
str. Separator for word splitting.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of words (or tokens).
</td>
</tr>

</table>

