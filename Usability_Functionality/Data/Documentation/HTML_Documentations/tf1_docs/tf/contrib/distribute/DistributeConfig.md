<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.distribute.DistributeConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="train_distribute"/>
<meta itemprop="property" content="eval_distribute"/>
<meta itemprop="property" content="remote_cluster"/>
</div>

# tf.contrib.distribute.DistributeConfig

## Class `DistributeConfig`

A config tuple for distribution strategies.



<!-- Placeholder for "Used in" -->


#### Attributes:


* <b>`train_distribute`</b>: a `DistributionStrategy` object for training.
* <b>`eval_distribute`</b>: an optional `DistributionStrategy` object for
  evaluation.
* <b>`remote_cluster`</b>: a dict, `ClusterDef` or `ClusterSpec` object specifying
  the cluster configurations. If this is given, the `train_and_evaluate`
  method will be running as a standalone client which connects to the
  cluster for training.

## Properties

<h3 id="train_distribute"><code>train_distribute</code></h3>




<h3 id="eval_distribute"><code>eval_distribute</code></h3>




<h3 id="remote_cluster"><code>remote_cluster</code></h3>






