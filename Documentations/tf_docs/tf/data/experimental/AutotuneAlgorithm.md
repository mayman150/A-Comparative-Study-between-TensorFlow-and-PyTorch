description: Represents the type of autotuning algorithm to use.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.AutotuneAlgorithm" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT"/>
<meta itemprop="property" content="GRADIENT_DESCENT"/>
<meta itemprop="property" content="HILL_CLIMB"/>
<meta itemprop="property" content="MAX_PARALLELISM"/>
<meta itemprop="property" content="STAGE_BASED"/>
</div>

# tf.data.experimental.AutotuneAlgorithm

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/ops/options.py">View source</a>



Represents the type of autotuning algorithm to use.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.AutotuneAlgorithm`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

DEFAULT: The default behavior is implementation specific and may change over
time.

HILL_CLIMB: In each optimization step, this algorithm chooses the optimial
parameter and increases its value by 1.

GRADIENT_DESCENT: In each optimization step, this algorithm updates the
parameter values in the optimal direction.

MAX_PARALLELISM: Similar to HILL_CLIMB but uses a relaxed stopping condition,
allowing the optimization to oversubscribe the CPU.

STAGE_BASED: In each optimization step, this algorithm chooses the worst
bottleneck parameter and increases its value by 1.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
DEFAULT<a id="DEFAULT"></a>
</td>
<td>
`<AutotuneAlgorithm.DEFAULT: 0>`
</td>
</tr><tr>
<td>
GRADIENT_DESCENT<a id="GRADIENT_DESCENT"></a>
</td>
<td>
`<AutotuneAlgorithm.GRADIENT_DESCENT: 2>`
</td>
</tr><tr>
<td>
HILL_CLIMB<a id="HILL_CLIMB"></a>
</td>
<td>
`<AutotuneAlgorithm.HILL_CLIMB: 1>`
</td>
</tr><tr>
<td>
MAX_PARALLELISM<a id="MAX_PARALLELISM"></a>
</td>
<td>
`<AutotuneAlgorithm.MAX_PARALLELISM: 3>`
</td>
</tr><tr>
<td>
STAGE_BASED<a id="STAGE_BASED"></a>
</td>
<td>
`<AutotuneAlgorithm.STAGE_BASED: 4>`
</td>
</tr>
</table>

