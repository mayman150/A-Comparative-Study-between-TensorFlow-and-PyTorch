description: Generate batches of tensor image data with real-time data augmentation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.ImageDataGenerator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_transform"/>
<meta itemprop="property" content="fit"/>
<meta itemprop="property" content="flow"/>
<meta itemprop="property" content="flow_from_dataframe"/>
<meta itemprop="property" content="flow_from_directory"/>
<meta itemprop="property" content="get_random_transform"/>
<meta itemprop="property" content="random_transform"/>
<meta itemprop="property" content="standardize"/>
</div>

# tf.keras.preprocessing.image.ImageDataGenerator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1166-L2145">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generate batches of tensor image data with real-time data augmentation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode=&#x27;nearest&#x27;,
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    interpolation_order=1,
    dtype=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/image/ImageDataGenerator.md"><code>tf.keras.preprocessing.image.ImageDataGenerator</code></a> is not
recommended for new code. Prefer loading images with
<a href="../../../../tf/keras/utils/image_dataset_from_directory.md"><code>tf.keras.utils.image_dataset_from_directory</code></a> and transforming the output
<a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> with preprocessing layers. For more information, see the
tutorials for [loading images](
https://www.tensorflow.org/tutorials/load_data/images) and
[augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
the [preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

 The data will be looped over (in batches).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`featurewise_center`<a id="featurewise_center"></a>
</td>
<td>
Boolean. Set input mean to 0 over the dataset,
feature-wise.
</td>
</tr><tr>
<td>
`samplewise_center`<a id="samplewise_center"></a>
</td>
<td>
Boolean. Set each sample mean to 0.
</td>
</tr><tr>
<td>
`featurewise_std_normalization`<a id="featurewise_std_normalization"></a>
</td>
<td>
Boolean. Divide inputs by std of the
dataset, feature-wise.
</td>
</tr><tr>
<td>
`samplewise_std_normalization`<a id="samplewise_std_normalization"></a>
</td>
<td>
Boolean. Divide each input by its std.
</td>
</tr><tr>
<td>
`zca_epsilon`<a id="zca_epsilon"></a>
</td>
<td>
epsilon for ZCA whitening. Default is 1e-6.
</td>
</tr><tr>
<td>
`zca_whitening`<a id="zca_whitening"></a>
</td>
<td>
Boolean. Apply ZCA whitening.
</td>
</tr><tr>
<td>
`rotation_range`<a id="rotation_range"></a>
</td>
<td>
Int. Degree range for random rotations.
</td>
</tr><tr>
<td>
`width_shift_range`<a id="width_shift_range"></a>
</td>
<td>
Float, 1-D array-like or int
- float: fraction of total width, if < 1, or pixels if >= 1.
- 1-D array-like: random elements from the array.
- int: integer number of pixels from interval `(-width_shift_range,
  +width_shift_range)` - With `width_shift_range=2` possible values
  are integers `[-1, 0, +1]`, same as with `width_shift_range=[-1,
  0, +1]`, while with `width_shift_range=1.0` possible values are
  floats in the interval [-1.0, +1.0).
</td>
</tr><tr>
<td>
`height_shift_range`<a id="height_shift_range"></a>
</td>
<td>
Float, 1-D array-like or int
- float: fraction of total height, if < 1, or pixels if >= 1.
- 1-D array-like: random elements from the array.
- int: integer number of pixels from interval `(-height_shift_range,
  +height_shift_range)` - With `height_shift_range=2` possible
  values are integers `[-1, 0, +1]`, same as with
  `height_shift_range=[-1, 0, +1]`, while with
  `height_shift_range=1.0` possible values are floats in the
  interval [-1.0, +1.0).
</td>
</tr><tr>
<td>
`brightness_range`<a id="brightness_range"></a>
</td>
<td>
Tuple or list of two floats. Range for picking a
brightness shift value from.
</td>
</tr><tr>
<td>
`shear_range`<a id="shear_range"></a>
</td>
<td>
Float. Shear Intensity (Shear angle in counter-clockwise
direction in degrees)
</td>
</tr><tr>
<td>
`zoom_range`<a id="zoom_range"></a>
</td>
<td>
Float or [lower, upper]. Range for random zoom. If a float,
`[lower, upper] = [1-zoom_range, 1+zoom_range]`.
</td>
</tr><tr>
<td>
`channel_shift_range`<a id="channel_shift_range"></a>
</td>
<td>
Float. Range for random channel shifts.
</td>
</tr><tr>
<td>
`fill_mode`<a id="fill_mode"></a>
</td>
<td>
One of {"constant", "nearest", "reflect" or "wrap"}. Default
is 'nearest'. Points outside the boundaries of the input are filled
according to the given mode:
  - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
  - 'nearest':  aaaaaaaa|abcd|dddddddd
  - 'reflect':  abcddcba|abcd|dcbaabcd
  - 'wrap':  abcdabcd|abcd|abcdabcd
</td>
</tr><tr>
<td>
`cval`<a id="cval"></a>
</td>
<td>
Float or Int. Value used for points outside the boundaries when
`fill_mode = "constant"`.
</td>
</tr><tr>
<td>
`horizontal_flip`<a id="horizontal_flip"></a>
</td>
<td>
Boolean. Randomly flip inputs horizontally.
</td>
</tr><tr>
<td>
`vertical_flip`<a id="vertical_flip"></a>
</td>
<td>
Boolean. Randomly flip inputs vertically.
</td>
</tr><tr>
<td>
`rescale`<a id="rescale"></a>
</td>
<td>
rescaling factor. If None or 0, no rescaling
is applied, otherwise we multiply the data by the value provided
(after applying all other transformations). Defaults to `None`.
</td>
</tr><tr>
<td>
`preprocessing_function`<a id="preprocessing_function"></a>
</td>
<td>
function that will be applied on each input. The
function will run after the image is resized and augmented.
  The function should take one argument: one image (Numpy tensor with
    rank 3), and should output a Numpy tensor with the same shape.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
Image data format, either "channels_first" or
"channels_last". "channels_last" mode means that the images should
have shape `(samples, height, width, channels)`, "channels_first" mode
means that the images should have shape `(samples, channels, height,
width)`. When unspecified, uses `image_data_format` value found in
your Keras config file at `~/.keras/keras.json` (if exists) else
'channels_last'. Defaults to "channels_last".
</td>
</tr><tr>
<td>
`validation_split`<a id="validation_split"></a>
</td>
<td>
Float. Fraction of images reserved for validation
(strictly between 0 and 1).
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If the value of the argument, `data_format` is other than
`"channels_last"` or `"channels_first"`.
</td>
</tr><tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If the value of the argument, `validation_split` > 1
or `validation_split` < 0.
</td>
</tr>
</table>



#### Examples:



Example of using `.flow(x, y)`:

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
model.fit(datagen.flow(x_train, y_train, batch_size=32,
         subset='training'),
         validation_data=datagen.flow(x_train, y_train,
         batch_size=8, subset='validation'),
         steps_per_epoch=len(x_train) / 32, epochs=epochs)
# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

Example of using `.flow_from_directory(directory)`:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
model.fit(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

Example of transforming images and masks together.

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
model.fit(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```

## Methods

<h3 id="apply_transform"><code>apply_transform</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1984-L2046">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply_transform(
    x, transform_parameters
)
</code></pre>

Applies a transformation to an image according to given parameters.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
3D tensor, single image.
</td>
</tr><tr>
<td>
`transform_parameters`
</td>
<td>
Dictionary with string - parameter pairs
describing the transformation.
Currently, the following parameters
from the dictionary are used:
- `'theta'`: Float. Rotation angle in degrees.
- `'tx'`: Float. Shift in the x direction.
- `'ty'`: Float. Shift in the y direction.
- `'shear'`: Float. Shear angle in degrees.
- `'zx'`: Float. Zoom in the x direction.
- `'zy'`: Float. Zoom in the y direction.
- `'flip_horizontal'`: Boolean. Horizontal flip.
- `'flip_vertical'`: Boolean. Vertical flip.
- `'channel_shift_intensity'`: Float. Channel shift intensity.
- `'brightness'`: Float. Brightness shift intensity.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A transformed version of the input (same shape).
</td>
</tr>

</table>



<h3 id="fit"><code>fit</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L2061-L2145">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fit(
    x, augment=False, rounds=1, seed=None
)
</code></pre>

Fits the data generator to some sample data.

This computes the internal data stats related to the
data-dependent transformations, based on an array of sample data.

Only required if `featurewise_center` or
`featurewise_std_normalization` or `zca_whitening` are set to True.

When `rescale` is set to a value, rescaling is applied to
sample data before computing the internal data stats.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Sample data. Should have rank 4.
In case of grayscale data,
the channels axis should have value 1, in case
of RGB data, it should have value 3, and in case
of RGBA data, it should have value 4.
</td>
</tr><tr>
<td>
`augment`
</td>
<td>
Boolean (default: False).
Whether to fit on randomly augmented samples.
</td>
</tr><tr>
<td>
`rounds`
</td>
<td>
Int (default: 1).
If using data augmentation (`augment=True`),
this is how many augmentation passes over the data to use.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Int (default: None). Random seed.
</td>
</tr>
</table>



<h3 id="flow"><code>flow</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1486-L1560">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flow(
    x,
    y=None,
    batch_size=32,
    shuffle=True,
    sample_weight=None,
    seed=None,
    save_to_dir=None,
    save_prefix=&#x27;&#x27;,
    save_format=&#x27;png&#x27;,
    ignore_class_split=False,
    subset=None
)
</code></pre>

Takes data & label arrays, generates batches of augmented data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input data. Numpy array of rank 4 or a tuple. If tuple, the first
element should contain the images and the second element another
numpy array or a list of numpy arrays that gets passed to the
output without any modifications. Can be used to feed the model
miscellaneous data along with the images. In case of grayscale
data, the channels axis of the image array should have value 1, in
case of RGB data, it should have value 3, and in case of RGBA
data, it should have value 4.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Labels.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Int (default: 32).
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
Boolean (default: True).
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Sample weights.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Int (default: None).
</td>
</tr><tr>
<td>
`save_to_dir`
</td>
<td>
None or str (default: None). This allows you to
optionally specify a directory to which to save the augmented
pictures being generated (useful for visualizing what you are
doing).
</td>
</tr><tr>
<td>
`save_prefix`
</td>
<td>
Str (default: `''`). Prefix to use for filenames of
saved pictures (only relevant if `save_to_dir` is set).
</td>
</tr><tr>
<td>
`save_format`
</td>
<td>
one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
"tif", "jpg" (only relevant if `save_to_dir` is set). Default:
"png".
</td>
</tr><tr>
<td>
`ignore_class_split`
</td>
<td>
Boolean (default: False), ignore difference
in number of classes in labels across train and validation
split (useful for non-classification tasks)
</td>
</tr><tr>
<td>
`subset`
</td>
<td>
Subset of data (`"training"` or `"validation"`) if
`validation_split` is set in `ImageDataGenerator`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An `Iterator` yielding tuples of `(x, y)`
where `x` is a numpy array of image data
(in the case of a single image input) or a list
of numpy arrays (in the case with
additional inputs) and `y` is a numpy array
of corresponding labels. If 'sample_weight' is not None,
the yielded tuples are of the form `(x, y, sample_weight)`.
If `y` is None, only the numpy array `x` is returned.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the Value of the argument, `subset` is other than
"training" or "validation".
</td>
</tr>
</table>



<h3 id="flow_from_dataframe"><code>flow_from_dataframe</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1670-L1829">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flow_from_dataframe(
    dataframe,
    directory=None,
    x_col=&#x27;filename&#x27;,
    y_col=&#x27;class&#x27;,
    weight_col=None,
    target_size=(256, 256),
    color_mode=&#x27;rgb&#x27;,
    classes=None,
    class_mode=&#x27;categorical&#x27;,
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix=&#x27;&#x27;,
    save_format=&#x27;png&#x27;,
    subset=None,
    interpolation=&#x27;nearest&#x27;,
    validate_filenames=True,
    **kwargs
)
</code></pre>

Takes the dataframe and the path to a directory + generates batches.

 The generated batches contain augmented/normalized data.

**A simple tutorial can be found **[here](
                            http://bit.ly/keras_flow_from_dataframe).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dataframe`
</td>
<td>
Pandas dataframe containing the filepaths relative to
`directory` (or absolute paths if `directory` is None) of the
images in a string column. It should include other column/s
depending on the `class_mode`:
- if `class_mode` is `"categorical"` (default value) it must
    include the `y_col` column with the class/es of each image.
    Values in column can be string/list/tuple if a single class
    or list/tuple if multiple classes.
- if `class_mode` is `"binary"` or `"sparse"` it must include
    the given `y_col` column with class values as strings.
- if `class_mode` is `"raw"` or `"multi_output"` it should
    contain the columns specified in `y_col`.
- if `class_mode` is `"input"` or `None` no extra column is
    needed.
</td>
</tr><tr>
<td>
`directory`
</td>
<td>
string, path to the directory to read images from. If
`None`, data in `x_col` column should be absolute paths.
</td>
</tr><tr>
<td>
`x_col`
</td>
<td>
string, column in `dataframe` that contains the filenames (or
absolute paths if `directory` is `None`).
</td>
</tr><tr>
<td>
`y_col`
</td>
<td>
string or list, column/s in `dataframe` that has the target
data.
</td>
</tr><tr>
<td>
`weight_col`
</td>
<td>
string, column in `dataframe` that contains the sample
weights. Default: `None`.
</td>
</tr><tr>
<td>
`target_size`
</td>
<td>
tuple of integers `(height, width)`, default: `(256,
256)`. The dimensions to which all images found will be resized.
</td>
</tr><tr>
<td>
`color_mode`
</td>
<td>
one of "grayscale", "rgb", "rgba". Default: "rgb".
Whether the images will be converted to have 1 or 3 color
channels.
</td>
</tr><tr>
<td>
`classes`
</td>
<td>
optional list of classes (e.g. `['dogs', 'cats']`). Default
is None. If not provided, the list of classes will be
automatically inferred from the `y_col`, which will map to the
label indices, will be alphanumeric). The dictionary containing
the mapping from class names to class indices can be obtained via
the attribute `class_indices`.
</td>
</tr><tr>
<td>
`class_mode`
</td>
<td>
one of "binary", "categorical", "input", "multi_output",
"raw", sparse" or None. Default: "categorical".
Mode for yielding the targets:
- `"binary"`: 1D numpy array of binary labels,
- `"categorical"`: 2D numpy array of one-hot encoded labels.
  Supports multi-label output.
- `"input"`: images identical to input images (mainly used to
  work with autoencoders),
- `"multi_output"`: list with the values of the different
  columns,
- `"raw"`: numpy array of values in `y_col` column(s),
- `"sparse"`: 1D numpy array of integer labels,
- `None`, no targets are returned (the generator will only yield
  batches of image data, which is useful to use in
  `model.predict()`).
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
size of the batches of data (default: 32).
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
whether to shuffle the data (default: True)
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
optional random seed for shuffling and transformations.
</td>
</tr><tr>
<td>
`save_to_dir`
</td>
<td>
None or str (default: None). This allows you to
optionally specify a directory to which to save the augmented
pictures being generated (useful for visualizing what you are
doing).
</td>
</tr><tr>
<td>
`save_prefix`
</td>
<td>
str. Prefix to use for filenames of saved pictures
(only relevant if `save_to_dir` is set).
</td>
</tr><tr>
<td>
`save_format`
</td>
<td>
one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
"tif", "jpg" (only relevant if `save_to_dir` is set). Default:
"png".
</td>
</tr><tr>
<td>
`subset`
</td>
<td>
Subset of data (`"training"` or `"validation"`) if
`validation_split` is set in `ImageDataGenerator`.
</td>
</tr><tr>
<td>
`interpolation`
</td>
<td>
Interpolation method used to resample the image if
the target size is different from that of the loaded image.
Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
supported. If PIL version 3.4.0 or newer is installed, `"box"` and
`"hamming"` are also supported. By default, `"nearest"` is used.
</td>
</tr><tr>
<td>
`validate_filenames`
</td>
<td>
Boolean, whether to validate image filenames in
`x_col`. If `True`, invalid images will be ignored. Disabling this
option can lead to speed-up in the execution of this function.
Defaults to `True`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
legacy arguments for raising deprecation warnings.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `DataFrameIterator` yielding tuples of `(x, y)`
where `x` is a numpy array containing a batch
of images with shape `(batch_size, *target_size, channels)`
and `y` is a numpy array of corresponding labels.
</td>
</tr>

</table>



<h3 id="flow_from_directory"><code>flow_from_directory</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1562-L1668">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flow_from_directory(
    directory,
    target_size=(256, 256),
    color_mode=&#x27;rgb&#x27;,
    classes=None,
    class_mode=&#x27;categorical&#x27;,
    batch_size=32,
    shuffle=True,
    seed=None,
    save_to_dir=None,
    save_prefix=&#x27;&#x27;,
    save_format=&#x27;png&#x27;,
    follow_links=False,
    subset=None,
    interpolation=&#x27;nearest&#x27;,
    keep_aspect_ratio=False
)
</code></pre>

Takes the path to a directory & generates batches of augmented data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`directory`
</td>
<td>
string, path to the target directory. It should contain
one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images
inside each of the subdirectories directory tree will be included
in the generator. See [this script](
https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
for more details.
</td>
</tr><tr>
<td>
`target_size`
</td>
<td>
Tuple of integers `(height, width)`. The dimensions to
which all images found will be resized. Defaults to `(256,256)`.
</td>
</tr><tr>
<td>
`color_mode`
</td>
<td>
One of "grayscale", "rgb", "rgba". Default: "rgb".
Whether the images will be converted to have 1, 3, or 4 channels.
</td>
</tr><tr>
<td>
`classes`
</td>
<td>
Optional list of class subdirectories (e.g. `['dogs',
'cats']`). Default: None. If not provided, the list of classes
will be automatically inferred from the subdirectory
names/structure under `directory`, where each subdirectory will be
treated as a different class (and the order of the classes, which
will map to the label indices, will be alphanumeric). The
dictionary containing the mapping from class names to class
indices can be obtained via the attribute `class_indices`.
</td>
</tr><tr>
<td>
`class_mode`
</td>
<td>
One of "categorical", "binary", "sparse",
"input", or None.
Determines the type of label arrays that are returned:
  - "categorical" will be 2D one-hot encoded labels,
  - "binary" will be 1D binary labels,
  - "sparse" will be 1D integer labels,
  - "input" will be images identical
    to input images (mainly used to work with autoencoders).
  - If None, no labels are returned
    (the generator will only yield batches of image data,
    which is useful to use with `model.predict_generator()`).
    Please note that in case of class_mode None,
    the data still needs to reside in a subdirectory
    of `directory` for it to work correctly.
  Defaults to "categorical".
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Size of the batches of data. Defaults to `32`.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
Whether to shuffle the data If `False`, sorts the
data in alphanumeric order. Defaults to `True`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Optional random seed for shuffling and transformations.
</td>
</tr><tr>
<td>
`save_to_dir`
</td>
<td>
None or str (default: None). This allows you to
optionally specify a directory to which to save the augmented
pictures being generated (useful for visualizing what you are
doing).
</td>
</tr><tr>
<td>
`save_prefix`
</td>
<td>
Str. Prefix to use for filenames of saved pictures
(only relevant if `save_to_dir` is set).
</td>
</tr><tr>
<td>
`save_format`
</td>
<td>
one of "png", "jpeg", "bmp", "pdf", "ppm", "gif",
"tif", "jpg" (only relevant if `save_to_dir` is set).
Defaults to "png".
</td>
</tr><tr>
<td>
`follow_links`
</td>
<td>
Whether to follow symlinks inside
class subdirectories. Defaults to `False`.
</td>
</tr><tr>
<td>
`subset`
</td>
<td>
Subset of data (`"training"` or `"validation"`) if
`validation_split` is set in `ImageDataGenerator`.
</td>
</tr><tr>
<td>
`interpolation`
</td>
<td>
Interpolation method used to resample the image if
the target size is different from that of the loaded image.
Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
supported. If PIL version 3.4.0 or newer is installed, `"box"` and
`"hamming"` are also supported. Defaults to `"nearest"`.
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `DirectoryIterator` yielding tuples of `(x, y)`
where `x` is a numpy array containing a batch
of images with shape `(batch_size, *target_size, channels)`
and `y` is a numpy array of corresponding labels.
</td>
</tr>

</table>



<h3 id="get_random_transform"><code>get_random_transform</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1893-L1982">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_random_transform(
    img_shape, seed=None
)
</code></pre>

Generates random parameters for a transformation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`img_shape`
</td>
<td>
Tuple of integers.
Shape of the image that is transformed.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Random seed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dictionary containing randomly chosen parameters describing the
transformation.
</td>
</tr>

</table>



<h3 id="random_transform"><code>random_transform</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L2048-L2059">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>random_transform(
    x, seed=None
)
</code></pre>

Applies a random transformation to an image.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
3D tensor, single image.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Random seed.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A randomly transformed version of the input (same shape).
</td>
</tr>

</table>



<h3 id="standardize"><code>standardize</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L1831-L1891">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>standardize(
    x
)
</code></pre>

Applies the normalization configuration in-place to a batch of inputs.

`x` is changed in-place since the function is mainly used internally
to standardize images and feed them to your network. If a copy of `x`
would be created instead it would have a significant performance cost.
If you want to apply this method without changing the input in-place
you can call the method creating a copy before:

standardize(np.copy(x))

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Batch of inputs to be normalized.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The inputs, normalized.
</td>
</tr>

</table>





