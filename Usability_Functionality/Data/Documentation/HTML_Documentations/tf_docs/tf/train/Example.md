description: An Example is a standard proto storing data for training and inference.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.Example" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.Example

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/example/example.proto">View source</a>



An `Example` is a standard proto storing data for training and inference.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.Example`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

It contains a key-value store <a href="../../tf/train/Example.md#features"><code>Example.features</code></a> where each key (string) maps
to a <a href="../../tf/train/Feature.md"><code>tf.train.Feature</code></a> message which contains a fixed-type list. This flexible
and compact format allows the storage of large amounts of typed data, but
requires that the data shape and use be determined by the configuration files
and parsers that are used to read and write this format (refer to
<a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a> for details).

```
>>> from google.protobuf import text_format
>>> example = text_format.Parse('''
...   features {
...     feature {key: "my_feature"
...              value {int64_list {value: [1, 2, 3, 4]}}}
...   }''',
...   tf.train.Example())
```

Use <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a> to extract tensors from a serialized `Example` proto:

```
>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {'my_feature': tf.io.RaggedFeature(dtype=tf.int64)})
{'my_feature': <tf.Tensor: shape=(4,), dtype=float32,
                           numpy=array([1, 2, 3, 4], dtype=int64)>}
```

While the list of keys, and the contents of each key _could_ be different for
every `Example`, TensorFlow expects a fixed list of keys, each with a fixed
`tf.dtype`. A conformant `Example` dataset obeys the following conventions:

  - If a Feature `K` exists in one example with data type `T`, it must be of
      type `T` in all other examples when present. It may be omitted.
  - The number of instances of Feature `K` list data may vary across examples,
      depending on the requirements of the model.
  - If a Feature `K` doesn't exist in an example, a `K`-specific default will be
      used, if configured.
  - If a Feature `K` exists in an example but contains no items, the intent
      is considered to be an empty tensor and no default will be used.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`features`<a id="features"></a>
</td>
<td>
`Features features`
</td>
</tr>
</table>



