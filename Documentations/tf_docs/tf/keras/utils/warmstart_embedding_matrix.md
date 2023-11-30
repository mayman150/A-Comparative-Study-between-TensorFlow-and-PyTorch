description: Warm start embedding matrix with changing vocab.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.warmstart_embedding_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.warmstart_embedding_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/layer_utils.py#L976-L1085">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Warm start embedding matrix with changing vocab.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.warmstart_embedding_matrix(
    base_vocabulary,
    new_vocabulary,
    base_embeddings,
    new_embeddings_initializer=&#x27;uniform&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This util can be used to warmstart the embedding layer matrix when
vocabulary changes between previously saved checkpoint and model.
Vocabulary change could mean, the size of the new vocab is different or the
vocabulary is reshuffled or new vocabulary has been added to old vocabulary.
If the vocabulary size changes, size of the embedding layer matrix also
changes. This util remaps the old vocabulary embeddings to the new embedding
layer matrix.

#### Example:


Here is an example that demonstrates how to use the
`warmstart_embedding_matrix` util.
```
>>> import keras
>>> vocab_base = tf.convert_to_tensor(["unk", "a", "b", "c"])
>>> vocab_new = tf.convert_to_tensor(
...        ["unk", "unk", "a", "b", "c", "d", "e"])
>>> vectorized_vocab_base = np.random.rand(vocab_base.shape[0], 3)
>>> vectorized_vocab_new = np.random.rand(vocab_new.shape[0], 3)
>>> warmstarted_embedding_matrix = warmstart_embedding_matrix(
...       base_vocabulary=vocab_base,
...       new_vocabulary=vocab_new,
...       base_embeddings=vectorized_vocab_base,
...       new_embeddings_initializer=keras.initializers.Constant(
...         vectorized_vocab_new))
```

Here is an example that demonstrates how to get vocabulary and embedding
weights from layers, use the `warmstart_embedding_matrix` util to remap the
layer embeddings and continue with model training.
```
# get old and new vocabulary by using layer.get_vocabulary()
# for example assume TextVectorization layer is used
base_vocabulary = old_text_vectorization_layer.get_vocabulary()
new_vocabulary = new_text_vectorization_layer.get_vocabulary()
# get previous embedding layer weights
embedding_weights_base = model.get_layer('embedding').get_weights()[0]
warmstarted_embedding = keras.utils.warmstart_embedding_matrix(
                              base_vocabulary,
                              new_vocabulary,
                              base_embeddings=embedding_weights_base,
                              new_embeddings_initializer="uniform")
updated_embedding_variable = tf.Variable(warmstarted_embedding)

# update embedding layer weights
model.layers[1].embeddings = updated_embedding_variable
model.fit(..)
# continue with model training

```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`base_vocabulary`<a id="base_vocabulary"></a>
</td>
<td>
The list of vocabulary terms that
the preexisting embedding matrix `base_embeddings` represents.
It can be either a 1D array/tensor or a tuple/list of vocabulary
terms (strings), or a path to a vocabulary text file. If passing a
 file path, the file should contain one line per term in the
 vocabulary.
</td>
</tr><tr>
<td>
`new_vocabulary`<a id="new_vocabulary"></a>
</td>
<td>
The list of vocabulary terms for the new vocabulary
(same format as above).
</td>
</tr><tr>
<td>
`base_embeddings`<a id="base_embeddings"></a>
</td>
<td>
NumPy array or tensor representing the preexisting
embedding matrix.
</td>
</tr><tr>
<td>
`new_embeddings_initializer`<a id="new_embeddings_initializer"></a>
</td>
<td>
Initializer for embedding vectors for
previously unseen terms to be added to the new embedding matrix (see
<a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>). new_embedding matrix
needs to be specified with "constant" initializer.
matrix. None means "uniform". Default value is None.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
tf.tensor of remapped embedding layer matrix
</td>
</tr>

</table>

