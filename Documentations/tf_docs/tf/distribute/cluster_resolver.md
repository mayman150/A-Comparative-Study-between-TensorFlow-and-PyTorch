description: Public API for tf._api.v2.distribute.cluster_resolver namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.cluster_resolver" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.distribute.cluster_resolver

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.distribute.cluster_resolver namespace



## Classes

[`class ClusterResolver`](../../tf/distribute/cluster_resolver/ClusterResolver.md): Abstract class for all implementations of ClusterResolvers.

[`class GCEClusterResolver`](../../tf/distribute/cluster_resolver/GCEClusterResolver.md): ClusterResolver for Google Compute Engine.

[`class KubernetesClusterResolver`](../../tf/distribute/cluster_resolver/KubernetesClusterResolver.md): ClusterResolver for Kubernetes.

[`class SimpleClusterResolver`](../../tf/distribute/cluster_resolver/SimpleClusterResolver.md): Simple implementation of ClusterResolver that accepts all attributes.

[`class SlurmClusterResolver`](../../tf/distribute/cluster_resolver/SlurmClusterResolver.md): ClusterResolver for system with Slurm workload manager.

[`class TFConfigClusterResolver`](../../tf/distribute/cluster_resolver/TFConfigClusterResolver.md): Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar.

[`class TPUClusterResolver`](../../tf/distribute/cluster_resolver/TPUClusterResolver.md): Cluster Resolver for Google Cloud TPUs.

[`class UnionResolver`](../../tf/distribute/cluster_resolver/UnionResolver.md): Performs a union on underlying ClusterResolvers.

