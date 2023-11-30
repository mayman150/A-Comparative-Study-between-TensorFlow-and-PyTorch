description: Iterator capable of reading images from a directory on disk.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.DirectoryIterator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="next"/>
<meta itemprop="property" content="on_epoch_end"/>
<meta itemprop="property" content="reset"/>
<meta itemprop="property" content="set_processing_attrs"/>
<meta itemprop="property" content="allowed_class_modes"/>
<meta itemprop="property" content="white_list_formats"/>
</div>

# tf.keras.preprocessing.image.DirectoryIterator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L449-L622">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Iterator capable of reading images from a directory on disk.

Inherits From: [`Iterator`](../../../../tf/keras/preprocessing/image/Iterator.md), [`Sequence`](../../../../tf/keras/utils/Sequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.DirectoryIterator(
    directory,
    image_data_generator,
    target_size=(256, 256),
    color_mode=&#x27;rgb&#x27;,
    classes=None,
    class_mode=&#x27;categorical&#x27;,
    batch_size=32,
    shuffle=True,
    seed=None,
    data_format=None,
    save_to_dir=None,
    save_prefix=&#x27;&#x27;,
    save_format=&#x27;png&#x27;,
    follow_links=False,
    subset=None,
    interpolation=&#x27;nearest&#x27;,
    keep_aspect_ratio=False,
    dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/image/DirectoryIterator.md"><code>tf.keras.preprocessing.image.DirectoryIterator</code></a> is not
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
`directory`<a id="directory"></a>
</td>
<td>
Path to the directory to read images from. Each subdirectory
in this directory will be considered to contain images from one class,
or alternatively you could specify class subdirectories via the
`classes` argument.
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
`target_size`<a id="target_size"></a>
</td>
<td>
tuple of integers, dimensions to resize input images to.
</td>
</tr><tr>
<td>
`color_mode`<a id="color_mode"></a>
</td>
<td>
One of `"rgb"`, `"rgba"`, `"grayscale"`. Color mode to read
images.
</td>
</tr><tr>
<td>
`classes`<a id="classes"></a>
</td>
<td>
Optional list of strings, names of subdirectories containing
images from each class (e.g. `["dogs", "cats"]`). It will be computed
automatically if not set.
</td>
</tr><tr>
<td>
`class_mode`<a id="class_mode"></a>
</td>
<td>
Mode for yielding the targets:
- `"binary"`: binary targets (if there are only two classes),
- `"categorical"`: categorical targets,
- `"sparse"`: integer targets,
- `"input"`: targets are images identical to input images (mainly
  used to work with autoencoders),
- `None`: no targets get yielded (only input images are yielded).
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
`interpolation`<a id="interpolation"></a>
</td>
<td>
Interpolation method used to resample the image if the
target size is different from that of the loaded image. Supported
methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3
or newer is installed, "lanczos" is also supported. If PIL version
3.4.0 or newer is installed, "box" and "hamming" are also supported.
By default, "nearest" is used.
</td>
</tr><tr>
<td>
`keep_aspect_ratio`<a id="keep_aspect_ratio"></a>
</td>
<td>
Boolean, whether to resize images to a target size
without aspect ratio distortion. The image is cropped in the center
with target aspect ratio before resizing.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Dtype to use for generated arrays.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`filepaths`<a id="filepaths"></a>
</td>
<td>
List of absolute paths to image files.
</td>
</tr><tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
Class labels of every observation.
</td>
</tr><tr>
<td>
`sample_weight`<a id="sample_weight"></a>
</td>
<td>

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




<h3 id="set_processing_attrs"><code>set_processing_attrs</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L266-L353">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_processing_attrs(
    image_data_generator,
    target_size,
    color_mode,
    data_format,
    save_to_dir,
    save_prefix,
    save_format,
    subset,
    interpolation,
    keep_aspect_ratio
)
</code></pre>

Sets attributes to use later for processing files into a batch.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`image_data_generator`
</td>
<td>
Instance of `ImageDataGenerator`
to use for random transformations and normalization.
</td>
</tr><tr>
<td>
`target_size`
</td>
<td>
tuple of integers, dimensions to resize input images
to.
</td>
</tr><tr>
<td>
`color_mode`
</td>
<td>
One of `"rgb"`, `"rgba"`, `"grayscale"`.
Color mode to read images.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
String, one of `channels_first`, `channels_last`.
</td>
</tr><tr>
<td>
`save_to_dir`
</td>
<td>
Optional directory where to save the pictures
being yielded, in a viewable format. This is useful
for visualizing the random transformations being
applied, for debugging purposes.
</td>
</tr><tr>
<td>
`save_prefix`
</td>
<td>
String prefix to use for saving sample
images (if `save_to_dir` is set).
</td>
</tr><tr>
<td>
`save_format`
</td>
<td>
Format to use for saving sample images
(if `save_to_dir` is set).
</td>
</tr><tr>
<td>
`subset`
</td>
<td>
Subset of data (`"training"` or `"validation"`) if
validation_split is set in ImageDataGenerator.
</td>
</tr><tr>
<td>
`interpolation`
</td>
<td>
Interpolation method used to resample the image if
the target size is different from that of the loaded image.
Supported methods are "nearest", "bilinear", and "bicubic". If
PIL version 1.1.3 or newer is installed, "lanczos" is also
supported. If PIL version 3.4.0 or newer is installed, "box" and
"hamming" are also supported. By default, "nearest" is used.
</td>
</tr><tr>
<td>
`keep_aspect_ratio`
</td>
<td>
Boolean, whether to resize images to a target
size without aspect ratio distortion. The image is cropped in
the center with target aspect ratio before resizing.
</td>
</tr>
</table>



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
allowed_class_modes<a id="allowed_class_modes"></a>
</td>
<td>
```
{
 'binary',
 'categorical',
 'input',
 'sparse',
 None
}
```
</td>
</tr><tr>
<td>
white_list_formats<a id="white_list_formats"></a>
</td>
<td>
`('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')`
</td>
</tr>
</table>

