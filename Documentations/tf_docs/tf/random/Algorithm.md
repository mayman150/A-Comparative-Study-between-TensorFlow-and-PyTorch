description: A random-number-generation (RNG) algorithm.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.random.Algorithm" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="AUTO_SELECT"/>
<meta itemprop="property" content="PHILOX"/>
<meta itemprop="property" content="THREEFRY"/>
</div>

# tf.random.Algorithm

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/random_ops_util.py">View source</a>



A random-number-generation (RNG) algorithm.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.random.experimental.Algorithm`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.random.Algorithm`, `tf.compat.v1.random.experimental.Algorithm`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Many random-number generators (e.g. the `alg` argument of
<a href="../../tf/random/Generator.md"><code>tf.random.Generator</code></a> and <a href="../../tf/random/stateless_uniform.md"><code>tf.random.stateless_uniform</code></a>) in TF allow
you to choose the algorithm used to generate the (pseudo-)random
numbers. You can set the algorithm to be one of the options below.

* `PHILOX`: The Philox algorithm introduced in the paper ["Parallel
  Random Numbers: As Easy as 1, 2,
  3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
* `THREEFRY`: The ThreeFry algorithm introduced in the paper
  ["Parallel Random Numbers: As Easy as 1, 2,
  3"](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).
* `AUTO_SELECT`: Allow TF to automatically select the algorithm
  depending on the accelerator device. Note that with this option,
  running the same TF program on different devices may result in
  different random numbers. Also note that TF may select an
  algorithm that is different from `PHILOX` and `THREEFRY`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
AUTO_SELECT<a id="AUTO_SELECT"></a>
</td>
<td>
`<Algorithm.AUTO_SELECT: 3>`
</td>
</tr><tr>
<td>
PHILOX<a id="PHILOX"></a>
</td>
<td>
`<Algorithm.PHILOX: 1>`
</td>
</tr><tr>
<td>
THREEFRY<a id="THREEFRY"></a>
</td>
<td>
`<Algorithm.THREEFRY: 2>`
</td>
</tr>
</table>

