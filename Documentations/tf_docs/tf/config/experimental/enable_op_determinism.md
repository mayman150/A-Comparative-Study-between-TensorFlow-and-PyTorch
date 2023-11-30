description: Configures TensorFlow ops to run deterministically.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental.enable_op_determinism" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental.enable_op_determinism

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Configures TensorFlow ops to run deterministically.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.experimental.enable_op_determinism()
</code></pre>



<!-- Placeholder for "Used in" -->

When op determinism is enabled, TensorFlow ops will be deterministic. This
means that if an op is run multiple times with the same inputs on the same
hardware, it will have the exact same outputs each time. This is useful for
debugging models. Note that determinism in general comes at the expense of
lower performance and so your model may run slower when op determinism is
enabled.

If you want your TensorFlow program to run deterministically, put the
following code near the start of your program.

```python
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
```

Calling <a href="../../../tf/keras/utils/set_random_seed.md"><code>tf.keras.utils.set_random_seed</code></a> sets the Python seed, the NumPy seed,
and the TensorFlow seed. Setting these seeds is necessary to ensure any random
numbers your program generates are also deterministic.

By default, op determinism is not enabled, so ops might return different
results when run with the same inputs. These differences are often caused by
the use of asynchronous threads within the op nondeterministically changing
the order in which floating-point numbers are added. Most of these cases of
nondeterminism occur on GPUs, which have thousands of hardware threads that
are used to run ops. Enabling determinism directs such ops to use a different
algorithm, one that does not use threads in a nondeterministic way.

Another potential source of nondeterminism is <a href="../../../tf/data.md"><code>tf.data</code></a> based data processing.
Typically, this can introduce nondeterminsm due to the use of parallelism in
methods such as <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> producing inputs or running stateful ops in a
nondeterministic order. Enabling determinism will remove such sources of
nondeterminism.

Enabling determinism will likely make your model or your <a href="../../../tf/data.md"><code>tf.data</code></a> data
processing slower. For example, <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> can become several orders of
magnitude slower when the map function has random ops or other stateful ops.
See the “Determinism and tf.data” section below for more details. In future
TensorFlow releases, we plan on improving the performance of determinism,
especially for common scenarios such as <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a>.

Certain ops will raise an `UnimplementedError` because they do not yet have a
deterministic implementation. Additionally, due to bugs, some ops might be
nondeterministic and not raise an `UnimplementedError`. If you encounter such
ops, please [file an issue](https://github.com/tensorflow/tensorflow/issues).

An example of enabling determinism follows. The
<a href="../../../tf/nn/softmax_cross_entropy_with_logits.md"><code>tf.nn.softmax_cross_entropy_with_logits</code></a> op is run multiple times and the
output is shown to be the same each time. This example would likely fail when
run on a GPU if determinism were not enabled, because
<a href="../../../tf/nn/softmax_cross_entropy_with_logits.md"><code>tf.nn.softmax_cross_entropy_with_logits</code></a> uses a nondeterministic algorithm on
GPUs by default.

```python
labels = tf.random.normal((1, 10000))
logits = tf.random.normal((1, 10000))
output = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                 logits=logits)
for _ in range(5):
  output2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                    logits=logits)
  tf.debugging.assert_equal(output, output2)
```

## Writing deterministic models

You can make your models deterministic by enabling op determinism. This
means that you can train a model and finish each run with exactly the same
trainable variables. This also means that the inferences of your
previously-trained model will be exactly the same on each run. Typically,
models can be made deterministic by simply setting the seeds and enabling
op determinism, as in the example above. However, to guarantee that your
model operates deterministically, you must meet all the following
requirements:

* Call <a href="../../../tf/config/experimental/enable_op_determinism.md"><code>tf.config.experimental.enable_op_determinism()</code></a>, as mentioned above.
* Reproducibly reset any pseudorandom number generators (PRNGs) you’re using,
  such as by setting the seeds for the default PRNGs in TensorFlow, Python,
  and NumPy, as mentioned above. Note that certain newer NumPy classes like
 ` numpy.random.default_rng` ignore the global NumPy seed, so a seed must be
  explicitly passed to such classes, if used.
* Use the same hardware configuration in every run.
* Use the same software environment in every run (OS, checkpoints, version of
  CUDA and TensorFlow, environmental variables, etc). Note that determinism is
  not guaranteed across different versions of TensorFlow.
* Do not use constructs outside TensorFlow that are nondeterministic, such as
  reading from `/dev/random` or using multiple threads/processes in ways that
  influence TensorFlow’s behavior.
* Ensure your input pipeline is deterministic. If you use <a href="../../../tf/data.md"><code>tf.data</code></a>, this is
  done automatically (at the expense of performance). See "Determinism and
  tf.data" below for more information.
* Do not use <a href="../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a> and
  <a href="../../../tf/distribute/experimental/ParameterServerStrategy.md"><code>tf.distribute.experimental.ParameterServerStrategy</code></a>, which can introduce
  nondeterminism. Besides ops (including <a href="../../../tf/data.md"><code>tf.data</code></a> ops), these are the only
  known potential sources of nondeterminism within TensorFlow, (if you
  find more, please file an issue). Note that <a href="../../../tf/compat/v1/Session.md"><code>tf.compat.v1.Session</code></a> is
  required to use the TF1 API, so determinism cannot be guaranteed when using
  the TF1 API.
* Do not use nondeterministic custom ops.

## Additional details on determinism

For stateful ops to be deterministic, the state of the system must be the same
every time the op is run. For example the output of <a href="../../../tf/Variable.md#sparse_read"><code>tf.Variable.sparse_read</code></a>
(obviously) depends on both the variable value and the `indices` function
parameter.  When determinism is enabled, the side effects of stateful ops are
deterministic.

TensorFlow’s random ops, such as <a href="../../../tf/random/normal.md"><code>tf.random.normal</code></a>, will raise a
`RuntimeError` if determinism is enabled and a seed has not been set. However,
attempting to generate nondeterministic random numbers using Python or NumPy
will not raise such errors. Make sure you remember to set the Python and NumPy
seeds. Calling <a href="../../../tf/keras/utils/set_random_seed.md"><code>tf.keras.utils.set_random_seed</code></a> is an easy way to set all
three seeds.

Note that latency, memory consumption, throughput, and other performance
characteristics are *not* made deterministic by enabling op determinism.
Only op outputs and side effects are made deterministic. Additionally, a model
may nondeterministically raise a <a href="../../../tf/errors/ResourceExhaustedError.md"><code>tf.errors.ResourceExhaustedError</code></a> from a
lack of memory due to the fact that memory consumption is nondeterministic.

## Determinism and tf.data

Enabling deterministic ops makes <a href="../../../tf/data.md"><code>tf.data</code></a> deterministic in several ways:

1. For dataset methods with a `deterministic` argument, such as <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a>
   and <a href="../../../tf/data/Dataset.md#batch"><code>Dataset.batch</code></a>, the `deterministic` argument is overridden to be
   `True` irrespective of its setting.
2. The `tf.data.Option.experimental_deterministic` option is overridden to be
   `True` irrespective of its setting..
3. In <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> and <a href="../../../tf/data/Dataset.md#interleave"><code>Dataset.interleave</code></a>, if the map or interleave
   function has stateful random ops or other stateful ops, the function will
   run serially instead of in parallel. This means the `num_parallel_calls`
   argument to `map` and `interleave` is effectively ignored.
4. Prefetching with <a href="../../../tf/data/Dataset.md#prefetch"><code>Dataset.prefetch</code></a> will be disabled if any function run
   as part of the input pipeline has certain stateful ops. Similarly, any
   dataset method with a `num_parallel_calls` argument will be made to run
   serially if any function in the input pipeline has such stateful ops.
   Legacy random ops such as <a href="../../../tf/random/normal.md"><code>tf.random.normal</code></a> will *not* cause such datasets
   to be changed, but most other stateful ops will.

Unfortunately, due to (3), performance can be greatly reduced when stateful
ops are used in <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> due to no longer running the map function in
parallel. A common example of stateful ops used in <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> are random
ops, such as <a href="../../../tf/random/normal.md"><code>tf.random.normal</code></a>, which are typically used for distortions. One
way to work around this is to use stateless random ops instead. Alternatively
you can hoist all random ops into its own separate <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> call, making
the original <a href="../../../tf/data/Dataset.md#map"><code>Dataset.map</code></a> call stateless and thus avoid the need to serialize
its execution.

(4) can also cause performance to be reduced, but occurs less frequently than
(3) because legacy random ops do not cause (4) to take effect. However, unlike
(3), when there are non-random stateful ops in a user-defined function, every
`map` and `interleave` dataset is affected, instead of just the `map` or
`interleave` dataset with the function that has stateful ops. Additionally,
`prefetch` datasets and any dataset with the `num_parallel_calls` argument are
also affected.