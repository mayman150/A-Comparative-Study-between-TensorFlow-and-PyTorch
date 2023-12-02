<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.eager" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="ASYNC"/>
<meta itemprop="property" content="DEVICE_PLACEMENT_EXPLICIT"/>
<meta itemprop="property" content="DEVICE_PLACEMENT_SILENT"/>
<meta itemprop="property" content="DEVICE_PLACEMENT_WARN"/>
<meta itemprop="property" content="SYNC"/>
</div>

# Module: tf.contrib.eager

TensorFlow Eager execution prototype.

<!-- Placeholder for "Used in" -->

EXPERIMENTAL: APIs here are unstable and likely to change without notice.

To use, at program startup, call <a href="../../tf/enable_eager_execution.md"><code>tf.compat.v1.enable_eager_execution()</code></a>.















## Modules

[`metrics`](../../tf/contrib/eager/metrics.md) module: Metrics namespace.

## Classes

[`class Checkpoint`](../../tf/train/Checkpoint.md): Groups trackable objects, saving and restoring them.

[`class Checkpointable`](../../tf/contrib/checkpoint/Checkpointable.md): Manages dependencies on other objects.

[`class EagerVariableStore`](../../tf/contrib/eager/EagerVariableStore.md): Wrapper allowing functional layers to be used with eager execution.

[`class ExecutionCallback`](../../tf/contrib/eager/ExecutionCallback.md): Valid callback actions.

[`class GradientTape`](../../tf/GradientTape.md): Record operations for automatic differentiation.

[`class Iterator`](../../tf/contrib/eager/Iterator.md): An iterator producing tf.Tensor objects from a tf.data.Dataset.

[`class Network`](../../tf/contrib/eager/Network.md): Represents the composition of a set of Layers.

[`class Saver`](../../tf/contrib/eager/Saver.md): A tf.compat.v1.train.Saver adapter for use when eager execution is enabled.

[`class Sequential`](../../tf/contrib/eager/Sequential.md): Represents a linear sequence of Layers or functions.

[`class TensorSpec`](../../tf/TensorSpec.md): Describes a tf.Tensor.

[`class Variable`](../../tf/contrib/eager/Variable.md): Variable based on resource handles.

## Functions

[`add_execution_callback(...)`](../../tf/contrib/eager/add_execution_callback.md): Add an execution callback to the default eager context.

[`async_clear_error(...)`](../../tf/contrib/eager/async_clear_error.md): Clears errors raised during ASYNC execution mode.

[`async_wait(...)`](../../tf/contrib/eager/async_wait.md): Waits for ops dispatched in ASYNC mode to finish.

[`clear_execution_callbacks(...)`](../../tf/contrib/eager/clear_execution_callbacks.md): Clear all execution callbacks from the default eager context.

[`connect_to_remote_host(...)`](../../tf/config/experimental_connect_to_host.md): Connects to a single machine to enable remote execution on it.

[`custom_gradient(...)`](../../tf/custom_gradient.md): Decorator to define a function with a custom gradient.

[`defun(...)`](../../tf/contrib/eager/defun.md): Compiles a Python function into a callable TensorFlow graph.

[`enable_eager_execution(...)`](../../tf/enable_eager_execution.md): Enables eager execution for the lifetime of this program.

[`enable_remote_eager_execution(...)`](../../tf/contrib/eager/enable_remote_eager_execution.md): Enables eager execution for the lifetime of this program.

[`errstate(...)`](../../tf/contrib/eager/errstate.md): Context manager setting error state.

[`executing_eagerly(...)`](../../tf/executing_eagerly.md): Returns True if the current thread has eager execution enabled.

[`execution_mode(...)`](../../tf/contrib/eager/execution_mode.md): Context manager for setting execution mode for current thread.

[`function(...)`](../../tf/function.md): Creates a callable TensorFlow graph from a Python function.

[`get_optimizer_variables(...)`](../../tf/contrib/eager/get_optimizer_variables.md): Returns a list of variables for the given <a href="../../tf/train/Optimizer.md"><code>tf.compat.v1.train.Optimizer</code></a>.

[`gradients_function(...)`](../../tf/contrib/eager/gradients_function.md): Returns a function which differentiates f with respect to params.

[`implicit_gradients(...)`](../../tf/contrib/eager/implicit_gradients.md): Returns a function which differentiates f with respect to variables.

[`implicit_value_and_gradients(...)`](../../tf/contrib/eager/implicit_value_and_gradients.md): Returns a function which differentiates f with respect to variables.

[`in_eager_mode(...)`](../../tf/executing_eagerly.md): Returns True if the current thread has eager execution enabled.

[`inf_callback(...)`](../../tf/contrib/eager/inf_callback.md): A specialization of `inf_nan_callback` that checks for `inf`s only.

[`inf_nan_callback(...)`](../../tf/contrib/eager/inf_nan_callback.md): An execution callback that checks for `inf`s and `nan`s in output tensors.

[`list_devices(...)`](../../tf/config/experimental_list_devices.md): List the names of the available devices.

[`make_template(...)`](../../tf/contrib/eager/make_template.md): Make a template, optionally compiling func_ into a graph function.

[`nan_callback(...)`](../../tf/contrib/eager/nan_callback.md): A specialization of `inf_nan_callback` that checks for `nan`s only.

[`num_gpus(...)`](../../tf/contrib/eager/num_gpus.md): Get the number of available GPU devices.

[`py_func(...)`](../../tf/py_function.md): Wraps a python function into a TensorFlow op that executes it eagerly.

[`restore_network_checkpoint(...)`](../../tf/contrib/eager/restore_network_checkpoint.md): Restore the Network from a checkpoint. (deprecated)

[`restore_variables_on_create(...)`](../../tf/contrib/eager/restore_variables_on_create.md): ContextManager that restores variables on creation.

[`run(...)`](../../tf/contrib/eager/run.md): Runs the program with an optional main function and argv list.

[`run_all_tests_in_graph_and_eager_modes(...)`](../../tf/contrib/eager/run_all_tests_in_graph_and_eager_modes.md): Execute all test methods in the given class with and without eager.

[`run_test_in_graph_and_eager_modes(...)`](../../tf/contrib/eager/run_test_in_graph_and_eager_modes.md): Execute the decorated test with and without enabling eager execution.

[`save_network_checkpoint(...)`](../../tf/contrib/eager/save_network_checkpoint.md): Save variables from the Network to a checkpoint. (deprecated)

[`set_execution_mode(...)`](../../tf/contrib/eager/set_execution_mode.md): Sets execution mode for the current thread.

[`set_server_def(...)`](../../tf/contrib/eager/set_server_def.md)

[`seterr(...)`](../../tf/contrib/eager/seterr.md): Set how abnormal conditions are handled by the default eager context.

[`value_and_gradients_function(...)`](../../tf/contrib/eager/value_and_gradients_function.md): Returns a function that computes f and its derivative w.r.t. params.

## Other Members

* `ASYNC = 1` <a id="ASYNC"></a>
* `DEVICE_PLACEMENT_EXPLICIT = 0` <a id="DEVICE_PLACEMENT_EXPLICIT"></a>
* `DEVICE_PLACEMENT_SILENT = 2` <a id="DEVICE_PLACEMENT_SILENT"></a>
* `DEVICE_PLACEMENT_WARN = 1` <a id="DEVICE_PLACEMENT_WARN"></a>
* `SYNC = 0` <a id="SYNC"></a>
