description: Applies an affine transformation specified by the parameters given.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.image.apply_affine_transform" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.image.apply_affine_transform

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/image.py#L2482-L2623">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies an affine transformation specified by the parameters given.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.image.apply_affine_transform(
    x,
    theta=0,
    tx=0,
    ty=0,
    shear=0,
    zx=1,
    zy=1,
    row_axis=1,
    col_axis=2,
    channel_axis=0,
    fill_mode=&#x27;nearest&#x27;,
    cval=0.0,
    order=1
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
3D numpy array - a 2D image with one or more channels.
</td>
</tr><tr>
<td>
`theta`<a id="theta"></a>
</td>
<td>
Rotation angle in degrees.
</td>
</tr><tr>
<td>
`tx`<a id="tx"></a>
</td>
<td>
Width shift.
</td>
</tr><tr>
<td>
`ty`<a id="ty"></a>
</td>
<td>
Heigh shift.
</td>
</tr><tr>
<td>
`shear`<a id="shear"></a>
</td>
<td>
Shear angle in degrees.
</td>
</tr><tr>
<td>
`zx`<a id="zx"></a>
</td>
<td>
Zoom in x direction.
</td>
</tr><tr>
<td>
`zy`<a id="zy"></a>
</td>
<td>
Zoom in y direction
</td>
</tr><tr>
<td>
`row_axis`<a id="row_axis"></a>
</td>
<td>
Index of axis for rows (aka Y axis) in the input
image. Direction: left to right.
</td>
</tr><tr>
<td>
`col_axis`<a id="col_axis"></a>
</td>
<td>
Index of axis for columns (aka X axis) in the input
image. Direction: top to bottom.
</td>
</tr><tr>
<td>
`channel_axis`<a id="channel_axis"></a>
</td>
<td>
Index of axis for channels in the input image.
</td>
</tr><tr>
<td>
`fill_mode`<a id="fill_mode"></a>
</td>
<td>
Points outside the boundaries of the input
are filled according to the given mode
(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
</td>
</tr><tr>
<td>
`cval`<a id="cval"></a>
</td>
<td>
Value used for points outside the boundaries
of the input if `mode='constant'`.
</td>
</tr><tr>
<td>
`order`<a id="order"></a>
</td>
<td>
int, order of interpolation
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The transformed version of the input.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ImportError`<a id="ImportError"></a>
</td>
<td>
if SciPy is not available.
</td>
</tr>
</table>

