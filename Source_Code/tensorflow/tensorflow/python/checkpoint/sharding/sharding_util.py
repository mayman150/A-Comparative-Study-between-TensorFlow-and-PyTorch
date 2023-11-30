# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data structures and utilities for checkpoint sharding."""

import dataclasses
from typing import Callable, Mapping, Sequence

from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object


TensorSlice = Mapping[tensor_spec.TensorSpec, tensor_lib.Tensor]
TensorSliceDict = Mapping[str, TensorSlice]


@dataclasses.dataclass(frozen=True)
class ShardableTensor:
  """Tensor wrapper containing data necessary for sharding."""
  _tensor_save_spec: saveable_object.SaveSpec
  tensor: tensor_lib.Tensor
  dtype: dtypes.DType
  device: device_lib.DeviceSpec
  name: str
  shape: tensor_shape.TensorShape
  slice_spec: variables.Variable.SaveSliceInfo
  checkpoint_key: str
  trackable: base.Trackable

  def __hash__(self) -> int:
    return hash((self.name, self.dtype, str(self.device), self.checkpoint_key))


@dataclasses.dataclass(frozen=True)
class ShardingCallback:
  """Checkpoint sharding callback function, along with a text description."""
  callback: Callable[
      [Sequence[ShardableTensor], ...],
      Sequence[Mapping[
          str, Mapping[tensor_spec.TensorSpec, saveable_object.SaveSpec]]]]
  description: str

  def __hash__(self) -> int:
    if hasattr(self.callback, "__name__"):
      callback_hash = hash((self.callback.__module__, self.callback.__name__))
    else:
      callback_hash = id(self.callback)
    return hash((callback_hash, self.description))


def validate_shards(
    shards: Sequence[TensorSliceDict],
    shardable_tensors: Sequence[ShardableTensor],
    callback_description: str
) -> None:
  """Validates shards generated by the sharding_callback."""
  unseen_tensor_dict = {
      (shardable_tensor.slice_spec,
       shardable_tensor.checkpoint_key): shardable_tensor.tensor
      for shardable_tensor in shardable_tensors
      if shardable_tensor.tensor is not None}
  seen_tensor_set = set()

  for shard_tensors in shards:
    task_tensor = None
    for checkpoint_key, tensor_slice_dict in shard_tensors.items():
      for slice_spec, shard_tensor in tensor_slice_dict.items():
        shard_tensor_id = (slice_spec, checkpoint_key)

        # Validate uniqueness.
        if shard_tensor_id in seen_tensor_set:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, multiple "
              "tensors with the same checkpoint key and slice spec were "
              "found:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n")

        # Validate no added tensors.
        if shard_tensor_id not in unseen_tensor_dict:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "not originally in the object graph was found in the "
              "checkpoint shards:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n")

        # Validate no shape change.
        target_shape = unseen_tensor_dict[shard_tensor_id].shape
        if shard_tensor.shape != target_shape:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "was found with an altered shape:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n"
              f"  original tensor_shape: {target_shape}\n"
              f"  new tensor_shape: {shard_tensor.shape}\n")

        # Validate no dtype change.
        target_dtype = unseen_tensor_dict[shard_tensor_id].dtype
        if shard_tensor.dtype != target_dtype:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "was found with an altered dtype:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n"
              f"  original tensor_dtype: {target_dtype}\n"
              f"  new tensor_dtype: {shard_tensor.dtype}\n")

        # Validate same task in shard.
        if task_tensor is None:
          task_tensor = ShardableTensor
          task_tensor.device = shard_tensor.device
          task_tensor.checkpoint_key = checkpoint_key
          task_tensor.slice_spec = slice_spec
        else:
          task1 = device_lib.DeviceSpec.from_string(task_tensor.device).task
          task2 = device_lib.DeviceSpec.from_string(shard_tensor.device).task
          if task1 != task2:
            raise RuntimeError(
                "After executing the checkpoint sharding callback, tensors "
                "with different tasks were found in the same shard:\n"
                f"  callback_description: {callback_description}\n"
                "  tensor #1:"
                f"    checkpoint_key: {task_tensor.checkpoint_key}\n"
                f"    slice_spec: {task_tensor.slice_spec}\n"
                f"    task: {task1}\n"
                "  tensor #2:"
                f"    checkpoint_key: {checkpoint_key}\n"
                f"    slice_spec: {slice_spec}\n"
                f"    task: {task2}\n")

        del unseen_tensor_dict[shard_tensor_id]
        seen_tensor_set.add(shard_tensor_id)

  # validate no tensor removal
  if unseen_tensor_dict:
    tensors_info = ""
    for slice_spec, ckpt_key in unseen_tensor_dict:
      tensors_info += "  tensor:\n"
      tensors_info += f"    checkpoint_key: {ckpt_key}\n"
      tensors_info += f"    slice_spec: {slice_spec}\n"
    raise RuntimeError(
        "After executing the checkpoint sharding callback, tensors in the "
        "object graph were not found in the checkpoint shards:\n"
        f"  callback_description: {callback_description}\n"
        f"{tensors_info}")
