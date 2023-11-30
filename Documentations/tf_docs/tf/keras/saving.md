<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.keras.saving

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>







## Classes

[`class custom_object_scope`](../../tf/keras/saving/custom_object_scope.md): Exposes custom classes/functions to Keras deserialization internals.

## Functions

[`deserialize_keras_object(...)`](../../tf/keras/saving/deserialize_keras_object.md): Retrieve the object by deserializing the config dict.

[`get_custom_objects(...)`](../../tf/keras/saving/get_custom_objects.md): Retrieves a live reference to the global dictionary of custom objects.

[`get_registered_name(...)`](../../tf/keras/saving/get_registered_name.md): Returns the name registered to an object within the Keras framework.

[`get_registered_object(...)`](../../tf/keras/saving/get_registered_object.md): Returns the class associated with `name` if it is registered with Keras.

[`load_model(...)`](../../tf/keras/saving/load_model.md): Loads a model saved via `model.save()`.

[`register_keras_serializable(...)`](../../tf/keras/saving/register_keras_serializable.md): Registers an object with the Keras serialization framework.

[`save_model(...)`](../../tf/keras/saving/save_model.md): Saves a model as a TensorFlow SavedModel or HDF5 file.

[`serialize_keras_object(...)`](../../tf/keras/saving/serialize_keras_object.md): Retrieve the config dict by serializing the Keras object.

