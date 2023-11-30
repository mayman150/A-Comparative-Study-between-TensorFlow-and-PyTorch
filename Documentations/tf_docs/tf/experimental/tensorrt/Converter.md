description: An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.tensorrt.Converter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="convert"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="summary"/>
</div>

# tf.experimental.tensorrt.Converter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/tensorrt/trt_convert.py">View source</a>



An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.tensorrt.Converter(
    input_saved_model_dir=None,
    input_saved_model_tags=None,
    input_saved_model_signature_key=None,
    use_dynamic_shape=None,
    dynamic_shape_profile_strategy=None,
    max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES,
    precision_mode=TrtPrecisionMode.FP32,
    minimum_segment_size=3,
    maximum_cached_engines=1,
    use_calibration=True,
    allow_build_at_runtime=True,
    conversion_params=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Windows support is provided experimentally. No guarantee is made regarding
functionality or engineering support. Use at your own risk.

There are several ways to run the conversion:

1. FP32/FP16 precision

   ```python
   params = tf.experimental.tensorrt.ConversionParams(
       precision_mode='FP16')
   converter = tf.experimental.tensorrt.Converter(
       input_saved_model_dir="my_dir", conversion_params=params)
   converter.convert()
   converter.save(output_saved_model_dir)
   ```

   In this case, no TRT engines will be built or saved in the converted
   SavedModel. But if input data is available during conversion, we can still
   build and save the TRT engines to reduce the cost during inference (see
   option 2 below).

2. FP32/FP16 precision with pre-built engines

   ```python
   params = tf.experimental.tensorrt.ConversionParams(
       precision_mode='FP16',
       # Set this to a large enough number so it can cache all the engines.
       maximum_cached_engines=16)
   converter = tf.experimental.tensorrt.Converter(
       input_saved_model_dir="my_dir", conversion_params=params)
   converter.convert()

   # Define a generator function that yields input data, and use it to execute
   # the graph to build TRT engines.
   def my_input_fn():
     for _ in range(num_runs):
       inp1, inp2 = ...
       yield inp1, inp2

   converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
   converter.save(output_saved_model_dir)  # Generated engines will be saved.
   ```

   In this way, one engine will be built/saved for each unique input shapes of
   the TRTEngineOp. This is good for applications that cannot afford building
   engines during inference but have access to input data that is similar to
   the one used in production (for example, that has the same input shapes).
   Also, the generated TRT engines is platform dependent, so we need to run
   `build()` in an environment that is similar to production (e.g. with
   same type of GPU).

3. INT8 precision and calibration with pre-built engines

   ```python
   params = tf.experimental.tensorrt.ConversionParams(
       precision_mode='INT8',
       # Currently only one INT8 engine is supported in this mode.
       maximum_cached_engines=1,
       use_calibration=True)
   converter = tf.experimental.tensorrt.Converter(
       input_saved_model_dir="my_dir", conversion_params=params)

   # Define a generator function that yields input data, and run INT8
   # calibration with the data. All input data should have the same shape.
   # At the end of convert(), the calibration stats (e.g. range information)
   # will be saved and can be used to generate more TRT engines with different
   # shapes. Also, one TRT engine will be generated (with the same shape as
   # the calibration data) for save later.
   def my_calibration_input_fn():
     for _ in range(num_runs):
       inp1, inp2 = ...
       yield inp1, inp2

   converter.convert(calibration_input_fn=my_calibration_input_fn)

   # (Optional) Generate more TRT engines offline (same as the previous
   # option), to avoid the cost of generating them during inference.
   def my_input_fn():
     for _ in range(num_runs):
       inp1, inp2 = ...
       yield inp1, inp2
   converter.build(input_fn=my_input_fn)

   # Save the TRT engine and the engines.
   converter.save(output_saved_model_dir)
   ```
4. To use dynamic shape, we need to call the build method with an input
   function to generate profiles. This step is similar to the INT8 calibration
   step described above. The converter also needs to be created with
   use_dynamic_shape=True and one of the following profile_strategies for
   creating profiles based on the inputs produced by the input function:
   * `Range`: create one profile that works for inputs with dimension values
     in the range of [min_dims, max_dims] where min_dims and max_dims are
     derived from the provided inputs.
   * `Optimal`: create one profile for each input. The profile only works for
     inputs with the same dimensions as the input it is created for. The GPU
     engine will be run with optimal performance with such inputs.
   * `Range+Optimal`: create the profiles for both `Range` and `Optimal`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_saved_model_dir`<a id="input_saved_model_dir"></a>
</td>
<td>
the directory to load the SavedModel which contains
the input graph to transforms. Required.
</td>
</tr><tr>
<td>
`input_saved_model_tags`<a id="input_saved_model_tags"></a>
</td>
<td>
list of tags to load the SavedModel.
</td>
</tr><tr>
<td>
`input_saved_model_signature_key`<a id="input_saved_model_signature_key"></a>
</td>
<td>
the key of the signature to optimize the
graph for.
</td>
</tr><tr>
<td>
`use_dynamic_shape`<a id="use_dynamic_shape"></a>
</td>
<td>
whether to enable dynamic shape support. None is
equivalent to False in the current implementation.
</td>
</tr><tr>
<td>
`dynamic_shape_profile_strategy`<a id="dynamic_shape_profile_strategy"></a>
</td>
<td>
one of the strings in
supported_profile_strategies(). None is equivalent to Range in the
current implementation.
</td>
</tr><tr>
<td>
`max_workspace_size_bytes`<a id="max_workspace_size_bytes"></a>
</td>
<td>
the maximum GPU temporary memory that the TRT
engine can use at execution time. This corresponds to the
'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
</td>
</tr><tr>
<td>
`precision_mode`<a id="precision_mode"></a>
</td>
<td>
one of the strings in
TrtPrecisionMode.supported_precision_modes().
</td>
</tr><tr>
<td>
`minimum_segment_size`<a id="minimum_segment_size"></a>
</td>
<td>
the minimum number of nodes required for a subgraph
to be replaced by TRTEngineOp.
</td>
</tr><tr>
<td>
`maximum_cached_engines`<a id="maximum_cached_engines"></a>
</td>
<td>
max number of cached TRT engines for dynamic TRT
ops. Created TRT engines for a dynamic dimension are cached. If the
number of cached engines is already at max but none of them supports the
input shapes, the TRTEngineOp will fall back to run the original TF
subgraph that corresponds to the TRTEngineOp.
</td>
</tr><tr>
<td>
`use_calibration`<a id="use_calibration"></a>
</td>
<td>
this argument is ignored if precision_mode is not INT8.
If set to True, a calibration graph will be created to calibrate the
missing ranges. The calibration graph must be converted to an inference
graph by running calibration with calibrate(). If set to False,
quantization nodes will be expected for every tensor in the graph
(excluding those which will be fused). If a range is missing, an error
will occur. Please note that accuracy may be negatively affected if
there is a mismatch between which tensors TRT quantizes and which
tensors were trained with fake quantization.
</td>
</tr><tr>
<td>
`allow_build_at_runtime`<a id="allow_build_at_runtime"></a>
</td>
<td>
whether to allow building TensorRT engines during
runtime if no prebuilt TensorRT engine can be found that can handle the
given inputs during runtime, then a new TensorRT engine is built at
runtime if allow_build_at_runtime=True, and otherwise native TF is used.
</td>
</tr><tr>
<td>
`conversion_params`<a id="conversion_params"></a>
</td>
<td>
a TrtConversionParams instance (deprecated).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if the combination of the parameters is invalid.
</td>
</tr>
</table>



## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/tensorrt/trt_convert.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    input_fn
)
</code></pre>

Run inference with converted graph in order to build TensorRT engines.

If the conversion requires INT8 calibration, then a reference to the
calibration function was stored during the call to convert(). Calibration
will be performed while we build the TensorRT engines.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_fn`
</td>
<td>
a generator function that provides the input data as a single
array, OR a list or tuple of the arrays OR a dict, which will be used
to execute the converted signature to generate TRT engines.
Example 1:
`def input_fn():
     # Let's assume a network with 1 input tensor.
     # We generate 2 sets of dummy input data:
     input_shapes = [(1, 16),    # 1st shape
                     (2, 32)]    # 2nd shape
     for shapes in input_shapes:
         # return an input tensor
         yield np.zeros(shape).astype(np.float32)'

Example 2:
`def input_fn():
     # Let's assume a network with 2 input tensors.
     # We generate 3 sets of dummy input data:
     input_shapes = [[(1, 16), (2, 16)], # 1st input list
                     [(2, 32), (4, 32)], # 2nd list of two tensors
                     [(4, 32), (8, 32)]] # 3rd input list
     for shapes in input_shapes:
         # return a list of input tensors
         yield [np.zeros(x).astype(np.float32) for x in shapes]`
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`NotImplementedError`
</td>
<td>
build() is already called.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
the input_fx is None.
</td>
</tr>
</table>



<h3 id="convert"><code>convert</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/tensorrt/trt_convert.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert(
    calibration_input_fn=None
)
</code></pre>

Convert the input SavedModel in 2.0 format.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`calibration_input_fn`
</td>
<td>
a generator function that yields input data as a
list or tuple or dict, which will be used to execute the converted
signature for calibration. All the returned input data should have the
same shape. Example: `def input_fn(): yield input1, input2, input3`

If dynamic_shape_mode==False, (or if the graph has static input shapes)
then we run calibration and build the calibrated engine during
conversion.

If dynamic_shape_mode==True (and the graph has any unknown input
shape), then the reference to calibration_input_fn is stored, and the
calibration is actually performed when we build the engine (see
build()).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the input combination is invalid.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The TF-TRT converted Function.
</td>
</tr>

</table>



<h3 id="save"><code>save</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/tensorrt/trt_convert.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    output_saved_model_dir, save_gpu_specific_engines=True, options=None
)
</code></pre>

Save the converted SavedModel.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`output_saved_model_dir`
</td>
<td>
directory to saved the converted SavedModel.
</td>
</tr><tr>
<td>
`save_gpu_specific_engines`
</td>
<td>
whether to save TRT engines that have been
built. When True, all engines are saved and when False, the engines
are not saved and will be rebuilt at inference time. By using
save_gpu_specific_engines=False after doing INT8 calibration, inference
can be done on different GPUs than the GPU that the model was calibrated
and saved on.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
<a href="../../../tf/saved_model/SaveOptions.md"><code>tf.saved_model.SaveOptions</code></a> object for configuring save options.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
if the needed calibration hasn't been done.
</td>
</tr>
</table>



<h3 id="summary"><code>summary</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/compiler/tensorrt/trt_convert.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>summary(
    line_length=160, detailed=True, print_fn=None
)
</code></pre>

This method describes the results of the conversion by TF-TRT.

It includes information such as the name of the engine, the number of nodes
per engine, the input and output dtype, along with the input shape of each
TRTEngineOp.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`line_length`
</td>
<td>
Default line length when printing on the console. Minimum 160
characters long.
</td>
</tr><tr>
<td>
`detailed`
</td>
<td>
Whether or not to show the nodes inside each TRTEngineOp.
</td>
</tr><tr>
<td>
`print_fn`
</td>
<td>
Print function to use. Defaults to `print`. It will be called on
each line of the summary. You can set it to a custom function in order
to capture the string summary.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
if the graph is not converted.
</td>
</tr>
</table>





