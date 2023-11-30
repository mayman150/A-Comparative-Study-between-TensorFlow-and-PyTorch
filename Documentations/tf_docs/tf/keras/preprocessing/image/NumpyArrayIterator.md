description: Iterator yielding data from a Numpy array.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.NumpyArrayIterator" />
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

# tf.keras.preprocessing.image.NumpyArrayIterator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L625-L826">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Iterator yielding data from a Numpy array.

Inherits From: [`Iterator`](../../../../tf/keras/preprocessing/image/Iterator.md), [`Sequence`](../../../../tf/keras/utils/Sequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.NumpyArrayIterator(
    x,
    y,
    image_data_generator,
    batch_size=32,
    shuffle=False,
    sample_weight=None,
    seed=None,
    data_format=None,
    save_to_dir=None,
    save_prefix=&#x27;&#x27;,
    save_format=&#x27;png&#x27;,
    subset=None,
    ignore_class_split=False,
    dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/image/NumpyArrayIterator.md"><code>tf.keras.preprocessing.image.NumpyArrayIterator</code></a> is not
recommended for new code. Prefer loading images with
<a href="../../../../tf/keras/utils/image_dataset_from_directory.md"><code>tf.keras.utils.image_dataset_from_directory</code></a> and transforming the output
<a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> with preprocessing layers. For more information, see the
tutorials for [loading images](
https://www.tensorflow.org/tutorials/load_data/images) and
[augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
the [preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
Numpy array of input data or tuple. If tuple, the second elements is
either another numpy array or a list of numpy arrays, each of which
gets passed through as an output without any modifications.
</td>
</tr><tr>
<td>
`y`<a id="y"></a>
</td>
<td>
Numpy array of targets data.
</td>
</tr><tr>
<td>
`image_data_generator`<a id="image_data_generator"></a>
</td>
<td>
Instance of `ImageDataGenerator` to use for random
transformations and normalization.
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
`sample_weight`<a id="sample_weight"></a>
</td>
<td>
Numpy array of sample weights.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Random seed for data shuffling.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
String, one of `channels_first`, `channels_last`.
</td>
</tr><tr>
<td>
`save_to_dir`<a id="save_to_dir"></a>
</td>
<td>
Optional directory where to save the pictures being
yielded, in a viewable format. This is useful for visualizing the
random transformations being applied, for debugging purposes.
</td>
</tr><tr>
<td>
`save_prefix`<a id="save_prefix"></a>
</td>
<td>
String prefix to use for saving sample images (if
`save_to_dir` is set).
</td>
</tr><tr>
<td>
`save_format`<a id="save_format"></a>
</td>
<td>
Format to use for saving sample images (if `save_to_dir` is
set).
</td>
</tr><tr>
<td>
`subset`<a id="subset"></a>
</td>
<td>
Subset of data (`"training"` or `"validation"`) if
validation_split is set in ImageDataGenerator.
</td>
</tr><tr>
<td>
`ignore_class_split`<a id="ignore_class_split"></a>
</td>
<td>
Boolean (default: False), ignore difference
in number of classes in labels across train and validation
split (useful for non-classification tasks)
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Dtype to use for the generated arrays.
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

