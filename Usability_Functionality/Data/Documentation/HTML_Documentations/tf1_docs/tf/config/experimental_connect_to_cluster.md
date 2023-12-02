<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental_connect_to_cluster" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental_connect_to_cluster

Connects to the given cluster.

### Aliases:

* `tf.compat.v1.config.experimental_connect_to_cluster`
* `tf.compat.v2.compat.v1.config.experimental_connect_to_cluster`
* `tf.compat.v2.config.experimental_connect_to_cluster`
* `tf.config.experimental_connect_to_cluster`

``` python
tf.config.experimental_connect_to_cluster(
    cluster_spec_or_resolver,
    job_name='localhost',
    task_index=0,
    protocol=None
)
```

<!-- Placeholder for "Used in" -->

Will make devices on the cluster available to use. Note that calling this more
than once will work, but will invalidate any tensor handles on the old remote
devices.

If the given local job name is not present in the cluster specification, it
will be automatically added, using an unused port on the localhost.

#### Args:


* <b>`cluster_spec_or_resolver`</b>: A `ClusterSpec` or `ClusterResolver` describing
  the cluster.
* <b>`job_name`</b>: The name of the local job.
* <b>`task_index`</b>: The local task index.
* <b>`protocol`</b>: The communication protocol, such as `"grpc"`. If unspecified, will
  use the default from `python/platform/remote_utils.py`.