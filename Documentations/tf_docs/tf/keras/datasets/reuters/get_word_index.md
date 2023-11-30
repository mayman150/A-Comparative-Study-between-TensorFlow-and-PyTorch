description: Retrieves a dict mapping words to their index in the Reuters dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.datasets.reuters.get_word_index" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.datasets.reuters.get_word_index

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/datasets/reuters.py#L164-L191">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Retrieves a dict mapping words to their index in the Reuters dataset.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.datasets.reuters.get_word_index(
    path=&#x27;reuters_word_index.json&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Actual word indices starts from 3, with 3 indices reserved for:
0 (padding), 1 (start), 2 (oov).

E.g. word index of 'the' is 1, but the in the actual training data, the
index of 'the' will be 1 + 3 = 4. Vice versa, to translate word indices in
training data back to words using this mapping, indices need to substract 3.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`<a id="path"></a>
</td>
<td>
where to cache the data (relative to `~/.keras/dataset`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The word index dictionary. Keys are word strings, values are their
index.
</td>
</tr>

</table>

