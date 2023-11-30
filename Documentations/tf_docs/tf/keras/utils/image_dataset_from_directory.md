description: Generates a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from image files in a directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.image_dataset_from_directory" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.image_dataset_from_directory

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/image_dataset.py#L30-L335">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generates a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from image files in a directory.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.preprocessing.image_dataset_from_directory`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.image_dataset_from_directory(
    directory,
    labels=&#x27;inferred&#x27;,
    label_mode=&#x27;int&#x27;,
    class_names=None,
    color_mode=&#x27;rgb&#x27;,
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation=&#x27;bilinear&#x27;,
    follow_links=False,
    crop_to_aspect_ratio=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

If your directory structure is:

```
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

Then calling `image_dataset_from_directory(main_directory,
labels='inferred')` will return a `tf.data.Dataset` that yields batches of
images from the subdirectories `class_a` and `class_b`, together with labels
0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

Supported image formats: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif`.
Animated gifs are truncated to the first frame.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`directory`<a id="directory"></a>
</td>
<td>
Directory where the data is located.
If `labels` is `"inferred"`, it should contain
subdirectories, each containing images for a class.
Otherwise, the directory structure is ignored.
</td>
</tr><tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
Either `"inferred"`
(labels are generated from the directory structure),
`None` (no labels),
or a list/tuple of integer labels of the same size as the number of
image files found in the directory. Labels should be sorted
according to the alphanumeric order of the image file paths
(obtained via `os.walk(directory)` in Python).
</td>
</tr><tr>
<td>
`label_mode`<a id="label_mode"></a>
</td>
<td>
String describing the encoding of `labels`. Options are:
- `"int"`: means that the labels are encoded as integers
    (e.g. for `sparse_categorical_crossentropy` loss).
- `"categorical"` means that the labels are
    encoded as a categorical vector
    (e.g. for `categorical_crossentropy` loss).
- `"binary"` means that the labels (there can be only 2)
    are encoded as `float32` scalars with values 0 or 1
    (e.g. for `binary_crossentropy`).
- `None` (no labels).
</td>
</tr><tr>
<td>
`class_names`<a id="class_names"></a>
</td>
<td>
Only valid if `labels` is `"inferred"`.
This is the explicit list of class names
(must match names of subdirectories). Used to control the order
of the classes (otherwise alphanumerical order is used).
</td>
</tr><tr>
<td>
`color_mode`<a id="color_mode"></a>
</td>
<td>
One of `"grayscale"`, `"rgb"`, `"rgba"`.
Defaults to `"rgb"`. Whether the images will be converted to
have 1, 3, or 4 channels.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Size of the batches of data.
If `None`, the data will not be batched
(the dataset will yield individual samples). Defaults to 32.
</td>
</tr><tr>
<td>
`image_size`<a id="image_size"></a>
</td>
<td>
Size to resize images to after they are read from disk,
specified as `(height, width)`.
Since the pipeline processes batches of images that must all have
the same size, this must be provided. Defaults to `(256, 256)`.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether to shuffle the data. Defaults to `True`.
If set to `False`, sorts the data in alphanumeric order.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Optional random seed for shuffling and transformations.
</td>
</tr><tr>
<td>
`validation_split`<a id="validation_split"></a>
</td>
<td>
Optional float between 0 and 1,
fraction of data to reserve for validation.
</td>
</tr><tr>
<td>
`subset`<a id="subset"></a>
</td>
<td>
Subset of the data to return.
One of `"training"`, `"validation"`, or `"both"`.
Only used if `validation_split` is set.
When `subset="both"`, the utility returns a tuple of two datasets
(the training and validation datasets respectively).
</td>
</tr><tr>
<td>
`interpolation`<a id="interpolation"></a>
</td>
<td>
String, the interpolation method used when
resizing images. Defaults to `"bilinear"`.
Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
`"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
</td>
</tr><tr>
<td>
`follow_links`<a id="follow_links"></a>
</td>
<td>
Whether to visit subdirectories pointed to by symlinks.
Defaults to `False`.
</td>
</tr><tr>
<td>
`crop_to_aspect_ratio`<a id="crop_to_aspect_ratio"></a>
</td>
<td>
If `True`, resize the images without aspect
ratio distortion. When the original aspect ratio differs from the
target aspect ratio, the output image will be cropped so as to
return the largest possible window in the image
(of size `image_size`) that matches the target aspect ratio. By
default (`crop_to_aspect_ratio=False`), aspect ratio may not be
preserved.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Legacy keyword arguments.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>


</table>


A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object.

- If `label_mode` is `None`, it yields `float32` tensors of shape
    `(batch_size, image_size[0], image_size[1], num_channels)`,
    encoding images (see below for rules regarding `num_channels`).
- Otherwise, it yields a tuple `(images, labels)`, where `images` has
    shape `(batch_size, image_size[0], image_size[1], num_channels)`,
    and `labels` follows the format described below.

Rules regarding labels format:

- if `label_mode` is `"int"`, the labels are an `int32` tensor of shape
    `(batch_size,)`.
- if `label_mode` is `"binary"`, the labels are a `float32` tensor of
    1s and 0s of shape `(batch_size, 1)`.
- if `label_mode` is `"categorical"`, the labels are a `float32` tensor
    of shape `(batch_size, num_classes)`, representing a one-hot
    encoding of the class index.

Rules regarding number of channels in the yielded images:

- if `color_mode` is `"grayscale"`,
    there's 1 channel in the image tensors.
- if `color_mode` is `"rgb"`,
    there are 3 channels in the image tensors.
- if `color_mode` is `"rgba"`,
    there are 4 channels in the image tensors.