<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.Glob" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.Glob

Returns a list of files that match the given pattern(s).

### Aliases:

* `tf.compat.v1.gfile.Glob`
* `tf.compat.v2.compat.v1.gfile.Glob`
* `tf.gfile.Glob`

``` python
tf.gfile.Glob(filename)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`filename`</b>: string or iterable of strings. The glob pattern(s).


#### Returns:

A list of strings containing filenames that match the given pattern(s).



#### Raises:


* <b>`errors.OpError`</b>: If there are filesystem / directory listing errors.