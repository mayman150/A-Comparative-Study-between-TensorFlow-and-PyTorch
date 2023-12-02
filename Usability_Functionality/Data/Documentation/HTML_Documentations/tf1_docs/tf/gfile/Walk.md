<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.Walk" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.Walk

Recursive directory tree generator for directories.

### Aliases:

* `tf.compat.v1.gfile.Walk`
* `tf.compat.v2.compat.v1.gfile.Walk`
* `tf.gfile.Walk`

``` python
tf.gfile.Walk(
    top,
    in_order=True
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`top`</b>: string, a Directory name
* <b>`in_order`</b>: bool, Traverse in order if True, post order if False.  Errors that
  happen while listing directories are ignored.


#### Yields:

Each yield is a 3-tuple:  the pathname of a directory, followed by lists of
all its subdirectories and leaf files.
(dirname, [subdirname, subdirname, ...], [filename, filename, ...])
as strings
