description: Mainly used as part of a <a href="../../tf/train/SequenceExample.md"><code>tf.train.SequenceExample</code></a>.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.FeatureList" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.FeatureList

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/example/feature.proto">View source</a>



Mainly used as part of a <a href="../../tf/train/SequenceExample.md"><code>tf.train.SequenceExample</code></a>.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.FeatureList`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Contains a list of <a href="../../tf/train/Feature.md"><code>tf.train.Feature</code></a>s.

The <a href="../../tf/train/SequenceExample.md"><code>tf.train.SequenceExample</code></a> proto can be thought of as a
proto implementation of the following python type:

```
# tf.train.Feature
Feature = Union[List[bytes],
                List[int64],
                List[float]]

# tf.train.FeatureList
FeatureList = List[Feature]

# tf.train.FeatureLists
FeatureLists = Dict[str, FeatureList]

class SequenceExample(typing.NamedTuple):
  context: Dict[str, Feature]
  feature_lists: FeatureLists
```

This proto implements the `List[Feature]` portion.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`feature`<a id="feature"></a>
</td>
<td>
`repeated Feature feature`
</td>
</tr>
</table>



