description: Public API for tf_estimator.python.estimator.api._v1.estimator namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.estimator" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.estimator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/api/_v1/estimator/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Public API for tf_estimator.python.estimator.api._v1.estimator namespace



## Classes

[`class BaselineClassifier`](../../../tf/compat/v1/estimator/BaselineClassifier.md): A classifier that can establish a simple baseline. (deprecated) (deprecated)

[`class BaselineEstimator`](../../../tf/compat/v1/estimator/BaselineEstimator.md): An estimator that can establish a simple baseline. (deprecated) (deprecated)

[`class BaselineRegressor`](../../../tf/compat/v1/estimator/BaselineRegressor.md): A regressor that can establish a simple baseline. (deprecated) (deprecated)

[`class BestExporter`](../../../tf/estimator/BestExporter.md): This class exports the serving graph and checkpoints of the best models. (deprecated)

[`class BinaryClassHead`](../../../tf/estimator/BinaryClassHead.md): Creates a `Head` for single label binary classification. (deprecated)

[`class CheckpointSaverHook`](../../../tf/estimator/CheckpointSaverHook.md): Saves checkpoints every N steps or seconds. (deprecated)

[`class CheckpointSaverListener`](../../../tf/estimator/CheckpointSaverListener.md): Interface for listeners that take action before or after checkpoint save. (deprecated)

[`class DNNClassifier`](../../../tf/compat/v1/estimator/DNNClassifier.md): A classifier for TensorFlow DNN models. (deprecated) (deprecated)

[`class DNNEstimator`](../../../tf/compat/v1/estimator/DNNEstimator.md): An estimator for TensorFlow DNN models with user-specified head. (deprecated) (deprecated)

[`class DNNLinearCombinedClassifier`](../../../tf/compat/v1/estimator/DNNLinearCombinedClassifier.md): An estimator for TensorFlow Linear and DNN joined classification models. (deprecated) (deprecated)

[`class DNNLinearCombinedEstimator`](../../../tf/compat/v1/estimator/DNNLinearCombinedEstimator.md): An estimator for TensorFlow Linear and DNN joined models with custom head. (deprecated) (deprecated)

[`class DNNLinearCombinedRegressor`](../../../tf/compat/v1/estimator/DNNLinearCombinedRegressor.md): An estimator for TensorFlow Linear and DNN joined models for regression. (deprecated) (deprecated)

[`class DNNRegressor`](../../../tf/compat/v1/estimator/DNNRegressor.md): A regressor for TensorFlow DNN models. (deprecated) (deprecated)

[`class Estimator`](../../../tf/compat/v1/estimator/Estimator.md): Estimator class to train and evaluate TensorFlow models. (deprecated)

[`class EstimatorSpec`](../../../tf/estimator/EstimatorSpec.md): Ops and objects returned from a `model_fn` and passed to an `Estimator`. (deprecated)

[`class EvalSpec`](../../../tf/estimator/EvalSpec.md): Configuration for the "eval" part for the `train_and_evaluate` call. (deprecated)

[`class Exporter`](../../../tf/estimator/Exporter.md): A class representing a type of model export. (deprecated)

[`class FeedFnHook`](../../../tf/estimator/FeedFnHook.md): Runs `feed_fn` and sets the `feed_dict` accordingly. (deprecated)

[`class FinalExporter`](../../../tf/estimator/FinalExporter.md): This class exports the serving graph and checkpoints at the end. (deprecated)

[`class FinalOpsHook`](../../../tf/estimator/FinalOpsHook.md): A hook which evaluates `Tensors` at the end of a session. (deprecated)

[`class GlobalStepWaiterHook`](../../../tf/estimator/GlobalStepWaiterHook.md): Delays execution until global step reaches `wait_until_step`. (deprecated)

[`class Head`](../../../tf/estimator/Head.md): Interface for the head/top of a model. (deprecated)

[`class LatestExporter`](../../../tf/estimator/LatestExporter.md): This class regularly exports the serving graph and checkpoints. (deprecated)

[`class LinearClassifier`](../../../tf/compat/v1/estimator/LinearClassifier.md): Linear classifier model. (deprecated) (deprecated)

[`class LinearEstimator`](../../../tf/compat/v1/estimator/LinearEstimator.md): An estimator for TensorFlow linear models with user-specified head. (deprecated) (deprecated)

[`class LinearRegressor`](../../../tf/compat/v1/estimator/LinearRegressor.md): An estimator for TensorFlow Linear regression problems. (deprecated) (deprecated)

[`class LoggingTensorHook`](../../../tf/estimator/LoggingTensorHook.md): Prints the given tensors every N local steps, every N seconds, or at end. (deprecated)

[`class LogisticRegressionHead`](../../../tf/estimator/LogisticRegressionHead.md): Creates a `Head` for logistic regression. (deprecated)

[`class ModeKeys`](../../../tf/estimator/ModeKeys.md): Standard names for Estimator model modes. (deprecated)

[`class MultiClassHead`](../../../tf/estimator/MultiClassHead.md): Creates a `Head` for multi class classification. (deprecated)

[`class MultiHead`](../../../tf/estimator/MultiHead.md): Creates a `Head` for multi-objective learning. (deprecated)

[`class MultiLabelHead`](../../../tf/estimator/MultiLabelHead.md): Creates a `Head` for multi-label classification. (deprecated)

[`class NanLossDuringTrainingError`](../../../tf/estimator/NanLossDuringTrainingError.md): DEPRECATED FUNCTION

[`class NanTensorHook`](../../../tf/estimator/NanTensorHook.md): Monitors the loss tensor and stops training if loss is NaN. (deprecated)

[`class PoissonRegressionHead`](../../../tf/estimator/PoissonRegressionHead.md): Creates a `Head` for poisson regression using <a href="../../../tf/nn/log_poisson_loss.md"><code>tf.nn.log_poisson_loss</code></a>. (deprecated)

[`class ProfilerHook`](../../../tf/estimator/ProfilerHook.md): Captures CPU/GPU profiling information every N steps or seconds. (deprecated)

[`class RegressionHead`](../../../tf/estimator/RegressionHead.md): Creates a `Head` for regression using the `mean_squared_error` loss. (deprecated)

[`class RunConfig`](../../../tf/estimator/RunConfig.md): This class specifies the configurations for an `Estimator` run. (deprecated)

[`class SecondOrStepTimer`](../../../tf/estimator/SecondOrStepTimer.md): Timer that triggers at most once every N seconds or once every N steps. (deprecated)

[`class SessionRunArgs`](../../../tf/estimator/SessionRunArgs.md): Represents arguments to be added to a `Session.run()` call. (deprecated)

[`class SessionRunContext`](../../../tf/estimator/SessionRunContext.md): Provides information about the `session.run()` call being made. (deprecated)

[`class SessionRunHook`](../../../tf/estimator/SessionRunHook.md): Hook to extend calls to MonitoredSession.run(). (deprecated)

[`class SessionRunValues`](../../../tf/estimator/SessionRunValues.md): Contains the results of `Session.run()`. (deprecated)

[`class StepCounterHook`](../../../tf/estimator/StepCounterHook.md): Hook that counts steps per second. (deprecated)

[`class StopAtStepHook`](../../../tf/estimator/StopAtStepHook.md): Hook that requests stop at a specified step. (deprecated)

[`class SummarySaverHook`](../../../tf/estimator/SummarySaverHook.md): Saves summaries every N steps. (deprecated)

[`class TrainSpec`](../../../tf/estimator/TrainSpec.md): Configuration for the "train" part for the `train_and_evaluate` call. (deprecated)

[`class VocabInfo`](../../../tf/estimator/VocabInfo.md): Vocabulary information for warm-starting. (deprecated)

[`class WarmStartSettings`](../../../tf/estimator/WarmStartSettings.md): Settings for warm-starting in `tf.estimator.Estimators`. (deprecated)

## Functions

[`add_metrics(...)`](../../../tf/estimator/add_metrics.md): Creates a new <a href="../../../tf/estimator/Estimator.md"><code>tf.estimator.Estimator</code></a> which has given metrics. (deprecated)

[`classifier_parse_example_spec(...)`](../../../tf/compat/v1/estimator/classifier_parse_example_spec.md): Generates parsing spec for tf.parse_example to be used with classifiers. (deprecated)

[`regressor_parse_example_spec(...)`](../../../tf/compat/v1/estimator/regressor_parse_example_spec.md): Generates parsing spec for tf.parse_example to be used with regressors. (deprecated)

[`train_and_evaluate(...)`](../../../tf/estimator/train_and_evaluate.md): Train and evaluate the `estimator`. (deprecated)

