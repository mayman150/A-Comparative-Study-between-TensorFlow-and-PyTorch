description: Public API for tf._api.v2.random.experimental namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.experimental" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.random.experimental

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.random.experimental namespace



## Classes

[`class Algorithm`](../../tf/random/Algorithm.md): A random-number-generation (RNG) algorithm.

[`class Generator`](../../tf/random/Generator.md): Random-number generator.

## Functions

[`create_rng_state(...)`](../../tf/random/create_rng_state.md): Creates a RNG state from an integer or a vector.

[`get_global_generator(...)`](../../tf/random/get_global_generator.md): Retrieves the global generator.

[`index_shuffle(...)`](../../tf/random/experimental/index_shuffle.md): Outputs the position of `index` in a permutation of `[0, ..., max_index]`.

[`set_global_generator(...)`](../../tf/random/set_global_generator.md): Replaces the global generator with another `Generator` object.

[`stateless_fold_in(...)`](../../tf/random/fold_in.md): Folds in data to an RNG seed to form a new RNG seed.

[`stateless_shuffle(...)`](../../tf/random/experimental/stateless_shuffle.md): Randomly and deterministically shuffles a tensor along its first dimension.

[`stateless_split(...)`](../../tf/random/split.md): Splits an RNG seed into `num` new seeds by adding a leading axis.

