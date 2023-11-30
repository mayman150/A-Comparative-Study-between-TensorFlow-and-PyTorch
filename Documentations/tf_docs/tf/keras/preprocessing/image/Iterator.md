description: Base class for image data iterators.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.Iterator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="next"/>
<meta itemprop="property" content="on_epoch_end"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="white_list_formats"/>
</div>

# tf.keras.preprocessing.image.Iterator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L58-L178">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base class for image data iterators.

Inherits From: [`Sequence`](../../../../tf/keras/utils/Sequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.Iterator(
    n, batch_size, shuffle, seed
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/image/Iterator.md"><code>tf.keras.preprocessing.image.Iterator</code></a> is not recommended for
new code. Prefer loading images with
<a href="../../../../tf/keras/utils/image_dataset_from_directory.md"><code>tf.keras.utils.image_dataset_from_directory</code></a> and transforming the output
<a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> with preprocessing layers. For more information, see the
tutorials for [loading images](
https://www.tensorflow.org/tutorials/load_data/images) and
[augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
the [preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

Every `Iterator` must implement the `_get_batches_of_transformed_samples`
method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`n`<a id="n"></a>
</td>
<td>
Integer, total number of samples in the dataset to loop over.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Integer, size of a batch.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Boolean, whether to shuffle the data between epochs.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Random seeding for data shuffling.
</td>
</tr>
</table>



## Methods

<h3 id="next"><code>next</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L158-L168">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next()
</code></pre>

For python 2.x.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The next batch.
</td>
</tr>

</table>



<h3 id="on_epoch_end"><code>on_epoch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L121-L122">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_end()
</code></pre>

Method called at the end of every epoch.


<h3 id="reset"><code>reset</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L124-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset()
</code></pre>




<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L101-L116">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    idx
)
</code></pre>

Gets batch at position `index`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`index`
</td>
<td>
position of the batch in the Sequence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A batch
</td>
</tr>

</table>



<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L150-L153">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Create a generator that iterate over the Sequence.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L118-L119">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Number of batch in the Sequence.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The number of batches in the Sequence.
</td>
</tr>

</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
white_list_formats<a id="white_list_formats"></a>
</td>
<td>
`('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')`
</td>
</tr>
</table>

