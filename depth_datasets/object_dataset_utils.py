# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import OpenEXR as exr

import Imath

def crop_to_square(tensor):
  assert len(tensor.shape) == 3
  _, h, w = tensor.shape
  if h == w:
    return tensor
  is_vertical = h > w
  assert tensor.shape[-2] % 2 == 0 and tensor.shape[-1] % 2 == 0, "Only implemented for even sized tensors"

  if is_vertical:
    crop = (h - w) // 2
    cropped_tensor = tensor[:, crop:-crop, :]
  else:
    crop = (w - h) // 2
    cropped_tensor = tensor[:, :, crop:-crop]

  return cropped_tensor

def open_exr_depth(depth_f):
  file = exr.InputFile(depth_f)
  # header = file.header()

  dw = file.header()['dataWindow']
  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

  Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)

  channel_str = file.channel('R', Float_Type)

  depth = np.array(np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1))

  return depth
