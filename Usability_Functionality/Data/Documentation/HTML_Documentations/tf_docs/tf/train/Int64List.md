description: Used in <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Holds a list of Int64s.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.Int64List" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.Int64List

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/example/feature.proto">View source</a>



Used in <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Holds a list of Int64s.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.Int64List`</p>
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

This proto implements the `List[int64]` portion.

```
>>> from google.protobuf import text_format
>>> example = text_format.Parse('''
...   features {
...     feature {key: "my_feature"
...              value {int64_list {value: [1, 2, 3, 4]}}}
...   }''',
...   tf.train.Example())
>>>
>>> example.features.feature['my_feature'].int64_list.value
[1, 2, 3, 4]
```

Use <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a> to extract tensors from a serialized `Example` proto:

```
>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {'my_feature': tf.io.RaggedFeature(dtype=tf.int64)})
{'my_feature': <tf.Tensor: shape=(4,), dtype=float32,
                           numpy=array([1, 2, 3, 4], dtype=int64)>}
```

See the [`tf.train.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample)
guide for usage details.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`value`<a id="value"></a>
</td>
<td>
`repeated int64 value`
</td>
</tr>
</table>



