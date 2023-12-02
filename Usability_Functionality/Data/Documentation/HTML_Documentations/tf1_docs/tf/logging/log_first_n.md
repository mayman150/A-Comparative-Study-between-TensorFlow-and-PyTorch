<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.logging.log_first_n" />
<meta itemprop="path" content="Stable" />
</div>

# tf.logging.log_first_n

Log 'msg % args' at level 'level' only first 'n' times.

### Aliases:

* `tf.compat.v1.logging.log_first_n`
* `tf.compat.v2.compat.v1.logging.log_first_n`
* `tf.logging.log_first_n`

``` python
tf.logging.log_first_n(
    level,
    msg,
    n,
    *args
)
```

<!-- Placeholder for "Used in" -->

Not threadsafe.

#### Args:


* <b>`level`</b>: The level at which to log.
* <b>`msg`</b>: The message to be logged.
* <b>`n`</b>: The number of times this should be called before it is logged.
* <b>`*args`</b>: The args to be substituted into the msg.