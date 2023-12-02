<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v2.strings.to_hash_bucket" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v2.strings.to_hash_bucket

Converts each string in the input Tensor to its hash mod by a number of buckets.

``` python
tf.compat.v2.strings.to_hash_bucket(
    input,
    num_buckets,
    name=None
)
```

<!-- Placeholder for "Used in" -->

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
<a href="../../../../tf/strings/to_hash_bucket_fast.md"><code>tf.strings.to_hash_bucket_fast()</code></a> or <a href="../../../../tf/strings/to_hash_bucket_strong.md"><code>tf.strings.to_hash_bucket_strong()</code></a>.

#### Args:


* <b>`input`</b>: A `Tensor` of type `string`.
* <b>`num_buckets`</b>: An `int` that is `>= 1`. The number of buckets.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `int64`.
