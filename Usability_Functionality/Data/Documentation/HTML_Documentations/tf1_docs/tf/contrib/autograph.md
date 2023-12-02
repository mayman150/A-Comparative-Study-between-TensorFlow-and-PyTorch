<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.autograph" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.contrib.autograph

This is the legacy module for AutoGraph, kept for backward compatibility.

<!-- Placeholder for "Used in" -->

New users should instead use `tensorflow.python.autograph`.

## Classes

[`class AutoGraphError`](../../tf/contrib/autograph/AutoGraphError.md): Base class for all AutoGraph exceptions.

[`class ConversionOptions`](../../tf/contrib/autograph/ConversionOptions.md): Immutable container for global conversion flags.

[`class Feature`](../../tf/autograph/experimental/Feature.md): This enumeration represents optional conversion options.

[`class StackTraceMapper`](../../tf/contrib/autograph/StackTraceMapper.md): Remaps generated code to code it originated from.

## Functions

[`convert(...)`](../../tf/contrib/autograph/convert.md): Decorator that compiles a function to use TensorFlow ops.

[`converted_call(...)`](../../tf/contrib/autograph/converted_call.md): Compiles a function call inline.

[`do_not_convert(...)`](../../tf/autograph/experimental/do_not_convert.md): Decorator that suppresses the conversion of a function.

[`set_element_type(...)`](../../tf/contrib/autograph/set_element_type.md): Indicates that the entity is expected hold items of specified type/shape.

[`stack(...)`](../../tf/contrib/autograph/stack.md): Stacks the input, if it admits the notion of stacking.

[`to_code(...)`](../../tf/compat/v2/autograph/to_code.md): Similar to `to_graph`, but returns Python source code as a string.

[`to_graph(...)`](../../tf/compat/v2/autograph/to_graph.md): Converts a Python entity into a TensorFlow graph.

