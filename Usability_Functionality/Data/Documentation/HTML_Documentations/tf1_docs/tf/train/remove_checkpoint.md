<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.remove_checkpoint" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.remove_checkpoint

Removes a checkpoint given by `checkpoint_prefix`. (deprecated)

### Aliases:

* `tf.compat.v1.train.remove_checkpoint`
* `tf.compat.v2.compat.v1.train.remove_checkpoint`
* `tf.train.remove_checkpoint`

``` python
tf.train.remove_checkpoint(
    checkpoint_prefix,
    checkpoint_format_version=tf.train.SaverDef.V2,
    meta_graph_suffix='meta'
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.

#### Args:


* <b>`checkpoint_prefix`</b>: The prefix of a V1 or V2 checkpoint. Typically the result
  of <a href="../../tf/train/Saver.md#save"><code>Saver.save()</code></a> or that of <a href="../../tf/train/latest_checkpoint.md"><code>tf.train.latest_checkpoint()</code></a>, regardless of
  sharded/non-sharded or V1/V2.
* <b>`checkpoint_format_version`</b>: <a href="../../tf/train/SaverDef.md#CheckpointFormatVersion"><code>SaverDef.CheckpointFormatVersion</code></a>, defaults to
  <a href="../../tf/train/SaverDef.md#V2"><code>SaverDef.V2</code></a>.
* <b>`meta_graph_suffix`</b>: Suffix for `MetaGraphDef` file. Defaults to 'meta'.