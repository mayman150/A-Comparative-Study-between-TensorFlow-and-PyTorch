<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.keras.utils

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>







## Modules

[`experimental`](../../tf/keras/utils/experimental.md) module

[`legacy`](../../tf/keras/utils/legacy.md) module

## Classes

[`class CustomObjectScope`](../../tf/keras/saving/custom_object_scope.md): Exposes custom classes/functions to Keras deserialization internals.

[`class FeatureSpace`](../../tf/keras/utils/FeatureSpace.md): One-stop utility for preprocessing and encoding structured data.

[`class GeneratorEnqueuer`](../../tf/keras/utils/GeneratorEnqueuer.md): Builds a queue out of a data generator.

[`class OrderedEnqueuer`](../../tf/keras/utils/OrderedEnqueuer.md): Builds a Enqueuer from a Sequence.

[`class Progbar`](../../tf/keras/utils/Progbar.md): Displays a progress bar.

[`class Sequence`](../../tf/keras/utils/Sequence.md): Base object for fitting to a sequence of data, such as a dataset.

[`class SequenceEnqueuer`](../../tf/keras/utils/SequenceEnqueuer.md): Base class to enqueue inputs.

[`class SidecarEvaluator`](../../tf/keras/utils/SidecarEvaluator.md): A class designed for a dedicated evaluator task.

[`class StepsPerExecutionTuner`](../../tf/keras/utils/StepsPerExecutionTuner.md): Steps per execution tuner class.

[`class TimedThread`](../../tf/keras/utils/TimedThread.md): Time-based interval Threads.

[`class custom_object_scope`](../../tf/keras/saving/custom_object_scope.md): Exposes custom classes/functions to Keras deserialization internals.

## Functions

[`array_to_img(...)`](../../tf/keras/utils/array_to_img.md): Converts a 3D Numpy array to a PIL Image instance.

[`audio_dataset_from_directory(...)`](../../tf/keras/utils/audio_dataset_from_directory.md): Generates a <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from audio files in a directory.

[`deserialize_keras_object(...)`](../../tf/keras/saving/deserialize_keras_object.md): Retrieve the object by deserializing the config dict.

[`disable_interactive_logging(...)`](../../tf/keras/utils/disable_interactive_logging.md): Turn off interactive logging.

[`enable_interactive_logging(...)`](../../tf/keras/utils/enable_interactive_logging.md): Turn on interactive logging.

[`get_custom_objects(...)`](../../tf/keras/saving/get_custom_objects.md): Retrieves a live reference to the global dictionary of custom objects.

[`get_file(...)`](../../tf/keras/utils/get_file.md): Downloads a file from a URL if it not already in the cache.

[`get_registered_name(...)`](../../tf/keras/saving/get_registered_name.md): Returns the name registered to an object within the Keras framework.

[`get_registered_object(...)`](../../tf/keras/saving/get_registered_object.md): Returns the class associated with `name` if it is registered with Keras.

[`get_source_inputs(...)`](../../tf/keras/utils/get_source_inputs.md): Returns the list of input tensors necessary to compute `tensor`.

[`image_dataset_from_directory(...)`](../../tf/keras/utils/image_dataset_from_directory.md): Generates a <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from image files in a directory.

[`img_to_array(...)`](../../tf/keras/utils/img_to_array.md): Converts a PIL Image instance to a Numpy array.

[`is_interactive_logging_enabled(...)`](../../tf/keras/utils/is_interactive_logging_enabled.md): Check if interactive logging is enabled.

[`load_img(...)`](../../tf/keras/utils/load_img.md): Loads an image into PIL format.

[`model_to_dot(...)`](../../tf/keras/utils/model_to_dot.md): Convert a Keras model to dot format.

[`normalize(...)`](../../tf/keras/utils/normalize.md): Normalizes a Numpy array.

[`pack_x_y_sample_weight(...)`](../../tf/keras/utils/pack_x_y_sample_weight.md): Packs user-provided data into a tuple.

[`pad_sequences(...)`](../../tf/keras/utils/pad_sequences.md): Pads sequences to the same length.

[`plot_model(...)`](../../tf/keras/utils/plot_model.md): Converts a Keras model to dot format and save to a file.

[`register_keras_serializable(...)`](../../tf/keras/saving/register_keras_serializable.md): Registers an object with the Keras serialization framework.

[`save_img(...)`](../../tf/keras/utils/save_img.md): Saves an image stored as a Numpy array to a path or file object.

[`serialize_keras_object(...)`](../../tf/keras/saving/serialize_keras_object.md): Retrieve the config dict by serializing the Keras object.

[`set_random_seed(...)`](../../tf/keras/utils/set_random_seed.md): Sets all random seeds for the program (Python, NumPy, and TensorFlow).

[`split_dataset(...)`](../../tf/keras/utils/split_dataset.md): Split a dataset into a left half and a right half (e.g. train / test).

[`text_dataset_from_directory(...)`](../../tf/keras/utils/text_dataset_from_directory.md): Generates a <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from text files in a directory.

[`timeseries_dataset_from_array(...)`](../../tf/keras/utils/timeseries_dataset_from_array.md): Creates a dataset of sliding windows over a timeseries provided as array.

[`to_categorical(...)`](../../tf/keras/utils/to_categorical.md): Converts a class vector (integers) to binary class matrix.

[`to_ordinal(...)`](../../tf/keras/utils/to_ordinal.md): Converts a class vector (integers) to an ordinal regression matrix.

[`unpack_x_y_sample_weight(...)`](../../tf/keras/utils/unpack_x_y_sample_weight.md): Unpacks user-provided data tuple.

[`warmstart_embedding_matrix(...)`](../../tf/keras/utils/warmstart_embedding_matrix.md): Warm start embedding matrix with changing vocab.

