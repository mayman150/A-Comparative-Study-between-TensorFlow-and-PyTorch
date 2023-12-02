<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.MakeDirs" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.MakeDirs

Creates a directory and all parent/intermediate directories.

### Aliases:

* `tf.compat.v1.gfile.MakeDirs`
* `tf.compat.v2.compat.v1.gfile.MakeDirs`
* `tf.gfile.MakeDirs`

``` python
tf.gfile.MakeDirs(dirname)
```

<!-- Placeholder for "Used in" -->

It succeeds if dirname already exists and is writable.

#### Args:


* <b>`dirname`</b>: string, name of the directory to be created


#### Raises:


* <b>`errors.OpError`</b>: If the operation fails.