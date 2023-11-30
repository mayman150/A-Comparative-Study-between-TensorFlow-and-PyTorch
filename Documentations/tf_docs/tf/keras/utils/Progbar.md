description: Displays a progress bar.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.Progbar" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add"/>
<meta itemprop="property" content="update"/>
</div>

# tf.keras.utils.Progbar

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/generic_utils.py#L128-L391">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Displays a progress bar.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.Progbar(
    target,
    width=30,
    verbose=1,
    interval=0.05,
    stateful_metrics=None,
    unit_name=&#x27;step&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`target`<a id="target"></a>
</td>
<td>
Total number of steps expected, None if unknown.
</td>
</tr><tr>
<td>
`width`<a id="width"></a>
</td>
<td>
Progress bar width on screen.
</td>
</tr><tr>
<td>
`verbose`<a id="verbose"></a>
</td>
<td>
Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
</td>
</tr><tr>
<td>
`stateful_metrics`<a id="stateful_metrics"></a>
</td>
<td>
Iterable of string names of metrics that should *not*
be averaged over time. Metrics in this list will be displayed as-is.
All others will be averaged by the progbar before display.
</td>
</tr><tr>
<td>
`interval`<a id="interval"></a>
</td>
<td>
Minimum visual progress update interval (in seconds).
</td>
</tr><tr>
<td>
`unit_name`<a id="unit_name"></a>
</td>
<td>
Display name for step counts (usually "step" or "sample").
</td>
</tr>
</table>



## Methods

<h3 id="add"><code>add</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/generic_utils.py#L331-L332">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add(
    n, values=None
)
</code></pre>




<h3 id="update"><code>update</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/generic_utils.py#L180-L329">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update(
    current, values=None, finalize=None
)
</code></pre>

Updates the progress bar.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`current`
</td>
<td>
Index of current step.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
List of tuples: `(name, value_for_last_step)`. If `name` is
in `stateful_metrics`, `value_for_last_step` will be displayed
as-is. Else, an average of the metric over time will be
displayed.
</td>
</tr><tr>
<td>
`finalize`
</td>
<td>
Whether this is the last update for the progress bar. If
`None`, uses `current >= self.target`. Defaults to `None`.
</td>
</tr>
</table>





