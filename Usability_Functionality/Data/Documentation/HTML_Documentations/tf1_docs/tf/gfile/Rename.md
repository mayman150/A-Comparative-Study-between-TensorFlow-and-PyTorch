<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.Rename" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.Rename

Rename or move a file / directory.

### Aliases:

* `tf.compat.v1.gfile.Rename`
* `tf.compat.v2.compat.v1.gfile.Rename`
* `tf.gfile.Rename`

``` python
tf.gfile.Rename(
    oldname,
    newname,
    overwrite=False
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`oldname`</b>: string, pathname for a file
* <b>`newname`</b>: string, pathname to which the file needs to be moved
* <b>`overwrite`</b>: boolean, if false it's an error for `newname` to be occupied by
  an existing file.


#### Raises:


* <b>`errors.OpError`</b>: If the operation fails.