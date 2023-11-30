description: Manages saving/restoring trackable values to disk, for DTensor. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.DTensorCheckpoint" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="read"/>
<meta itemprop="property" content="restore"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="sync"/>
<meta itemprop="property" content="write"/>
</div>

# tf.experimental.dtensor.DTensorCheckpoint

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/d_checkpoint.py">View source</a>



Manages saving/restoring trackable values to disk, for DTensor. (deprecated)

Inherits From: [`Checkpoint`](../../../tf/train/Checkpoint.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.DTensorCheckpoint(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    root=None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.train.Checkpoint instead of DTensorCheckpoint. DTensor is integrated with tf.train.Checkpoint and it can be used out of the box to save and restore dtensors.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`save_counter`<a id="save_counter"></a>
</td>
<td>
An integer variable which starts at zero and is incremented on save.

Used to number checkpoints.
</td>
</tr>
</table>



## Methods

<h3 id="read"><code>read</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>read(
    save_path, options=None
)
</code></pre>

Reads a training checkpoint written with `write`.

Reads this `Checkpoint` and any objects it depends on.

This method is just like `restore()` but does not expect the `save_counter`
variable in the checkpoint. It only restores the objects that the checkpoint
already depends on.

The method is primarily intended for use by higher level checkpoint
management utilities that use `write()` instead of `save()` and have their
own mechanisms to number and track checkpoints.

#### Example usage:



```python
# Create a checkpoint with write()
ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
path = ckpt.write('/tmp/my_checkpoint')

# Later, load the checkpoint with read()
# With restore() assert_consumed() would have failed.
checkpoint.read(path).assert_consumed()

# You can also pass options to read(). For example this
# runs the IO ops on the localhost:
options = tf.train.CheckpointOptions(
    experimental_io_device="/job:localhost")
checkpoint.read(path, options=options)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`save_path`
</td>
<td>
The path to the checkpoint as returned by `write`.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Optional <a href="../../../tf/train/CheckpointOptions.md"><code>tf.train.CheckpointOptions</code></a> object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A load status object, which can be used to make assertions about the
status of a checkpoint restoration.  See `restore` for details.
</td>
</tr>

</table>



<h3 id="restore"><code>restore</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>restore(
    save_path, options=None
)
</code></pre>

Restores a training checkpoint.

Restores this `Checkpoint` and any objects it depends on.

This method is intended to be used to load checkpoints created by `save()`.
For checkpoints created by `write()` use the `read()` method which does not
expect the `save_counter` variable added by `save()`.

`restore()` either assigns values immediately if variables to restore have
been created already, or defers restoration until the variables are
created. Dependencies added after this call will be matched if they have a
corresponding object in the checkpoint (the restore request will queue in
any trackable object waiting for the expected dependency to be added).

```python
checkpoint = tf.train.Checkpoint( ... )
checkpoint.restore(path)

# You can additionally pass options to restore():
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.restore(path, options=options)
```

To ensure that loading is complete and no more deferred restorations will
take place, use the `assert_consumed()` method of the status object returned
by `restore()`:

```python
checkpoint.restore(path, options=options).assert_consumed()
```

The assert will raise an error if any Python objects in the dependency graph
were not found in the checkpoint, or if any checkpointed values do not have
a matching Python object.

Name-based <a href="../../../tf/compat/v1/train/Saver.md"><code>tf.compat.v1.train.Saver</code></a> checkpoints from TensorFlow 1.x can be
loaded using this method. Names are used to match variables. Re-encode
name-based checkpoints using <a href="../../../tf/train/Checkpoint.md#save"><code>tf.train.Checkpoint.save</code></a> as soon as possible.

**Loading from SavedModel checkpoints**

To load values from a SavedModel, just pass the SavedModel directory
to checkpoint.restore:

```python
model = tf.keras.Model(...)
tf.saved_model.save(model, path)  # or model.save(path, save_format='tf')

checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(path).expect_partial()
```

This example calls `expect_partial()` on the loaded status, since
SavedModels saved from Keras often generates extra keys in the checkpoint.
Otherwise, the program prints a lot of warnings about unused keys at exit
time.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`save_path`
</td>
<td>
The path to the checkpoint, as returned by `save` or
<a href="../../../tf/train/latest_checkpoint.md"><code>tf.train.latest_checkpoint</code></a>. If the checkpoint was written by the
name-based <a href="../../../tf/compat/v1/train/Saver.md"><code>tf.compat.v1.train.Saver</code></a>, names are used to match
variables. This path may also be a SavedModel directory.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Optional <a href="../../../tf/train/CheckpointOptions.md"><code>tf.train.CheckpointOptions</code></a> object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A load status object, which can be used to make assertions about the
status of a checkpoint restoration.

The returned status object has the following methods:

* `assert_consumed()`:
    Raises an exception if any variables are unmatched: either
    checkpointed values which don't have a matching Python object or
    Python objects in the dependency graph with no values in the
    checkpoint. This method returns the status object, and so may be
    chained with other assertions.

* `assert_existing_objects_matched()`:
    Raises an exception if any existing Python objects in the dependency
    graph are unmatched. Unlike `assert_consumed`, this assertion will
    pass if values in the checkpoint have no corresponding Python
    objects. For example a `tf.keras.Layer` object which has not yet been
    built, and so has not created any variables, will pass this assertion
    but fail `assert_consumed`. Useful when loading part of a larger
    checkpoint into a new Python program, e.g. a training checkpoint with
    a <a href="../../../tf/compat/v1/train/Optimizer.md"><code>tf.compat.v1.train.Optimizer</code></a> was saved but only the state required
    for
    inference is being loaded. This method returns the status object, and
    so may be chained with other assertions.

* `assert_nontrivial_match()`: Asserts that something aside from the root
    object was matched. This is a very weak assertion, but is useful for
    sanity checking in library code where objects may exist in the
    checkpoint which haven't been created in Python and some Python
    objects may not have a checkpointed value.

* `expect_partial()`: Silence warnings about incomplete checkpoint
    restores. Warnings are otherwise printed for unused parts of the
    checkpoint file or object when the `Checkpoint` object is deleted
    (often at program shutdown).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`NotFoundError`
</td>
<td>
if the a checkpoint or SavedModel cannot be found at
`save_path`.
</td>
</tr>
</table>



<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    file_prefix, options=None
)
</code></pre>

Saves a training checkpoint and provides basic checkpoint management.

The saved checkpoint includes variables created by this object and any
trackable objects it depends on at the time <a href="../../../tf/train/Checkpoint.md#save"><code>Checkpoint.save()</code></a> is
called.

`save` is a basic convenience wrapper around the `write` method,
sequentially numbering checkpoints using `save_counter` and updating the
metadata used by <a href="../../../tf/train/latest_checkpoint.md"><code>tf.train.latest_checkpoint</code></a>. More advanced checkpoint
management, for example garbage collection and custom numbering, may be
provided by other utilities which also wrap `write` and `read`.
(<a href="../../../tf/train/CheckpointManager.md"><code>tf.train.CheckpointManager</code></a> for example).

```
step = tf.Variable(0, name="step")
checkpoint = tf.train.Checkpoint(step=step)
checkpoint.save("/tmp/ckpt")

# Later, read the checkpoint with restore()
checkpoint.restore("/tmp/ckpt-1")

# You can also pass options to save() and restore(). For example this
# runs the IO ops on the localhost:
options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.save("/tmp/ckpt", options=options)

# Later, read the checkpoint with restore()
checkpoint.restore("/tmp/ckpt-1", options=options)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_prefix`
</td>
<td>
A prefix to use for the checkpoint filenames
(/path/to/directory/and_a_prefix). Names are generated based on this
prefix and <a href="../../../tf/train/Checkpoint.md#save_counter"><code>Checkpoint.save_counter</code></a>.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Optional <a href="../../../tf/train/CheckpointOptions.md"><code>tf.train.CheckpointOptions</code></a> object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The full path to the checkpoint.
</td>
</tr>

</table>



<h3 id="sync"><code>sync</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sync()
</code></pre>

Wait for any outstanding save or restore operations.


<h3 id="write"><code>write</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write(
    file_prefix, options=None
)
</code></pre>

Writes a training checkpoint.

The checkpoint includes variables created by this object and any
trackable objects it depends on at the time <a href="../../../tf/train/Checkpoint.md#write"><code>Checkpoint.write()</code></a> is
called.

`write` does not number checkpoints, increment `save_counter`, or update the
metadata used by <a href="../../../tf/train/latest_checkpoint.md"><code>tf.train.latest_checkpoint</code></a>. It is primarily intended for
use by higher level checkpoint management utilities. `save` provides a very
basic implementation of these features.

Checkpoints written with `write` must be read with `read`.

#### Example usage:



```
step = tf.Variable(0, name="step")
checkpoint = tf.Checkpoint(step=step)
checkpoint.write("/tmp/ckpt")

# Later, read the checkpoint with read()
checkpoint.read("/tmp/ckpt")

# You can also pass options to write() and read(). For example this
# runs the IO ops on the localhost:
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.write("/tmp/ckpt", options=options)

# Later, read the checkpoint with read()
checkpoint.read("/tmp/ckpt", options=options)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`file_prefix`
</td>
<td>
A prefix to use for the checkpoint filenames
(/path/to/directory/and_a_prefix).
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Optional <a href="../../../tf/train/CheckpointOptions.md"><code>tf.train.CheckpointOptions</code></a> object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The full path to the checkpoint (i.e. `file_prefix`).
</td>
</tr>

</table>





