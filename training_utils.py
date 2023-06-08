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


import torch.optim
from lucent.optvis.param.images import *
from lucent.optvis.param.spatial import *
import torch.nn as nn

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.history = []
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    self.history.extend([float(val)] * n)


  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# replicates create_fft_image but into a class
class FFTImage(nn.Module):
  def __init__(self, w, h=None, sd=None, batch=None, decorrelate=True, channels=None, decay_power=1):
    super(FFTImage, self).__init__()

    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,)  # 2 for imaginary and real components
    sd = sd or 0.01

    self.spectrum_real_imag_t = nn.Parameter((torch.randn(*init_val_size) * sd))

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    self.scale = nn.Parameter(torch.tensor(scale).float()[None, None, ..., None])
    self.scale.requires_grad = False

    def inner():
      scaled_spectrum_t = self.scale * self.spectrum_real_imag_t
      import torch
      try:
        if type(scaled_spectrum_t) is not torch.complex64:
          scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
        image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
      except:
        image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
      image = image[:batch, :channels, :h, :w]
      # magic = 4.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
      # we replace the constant so at init it expands all the dynamic range
      magic = 0.2
      image = image / magic
      return image

    params, image_f = [self.spectrum_real_imag_t], inner
    if channels:
      output = to_valid_rgb(image_f, decorrelate=False)
    else:
      output = to_valid_rgb(image_f, decorrelate=decorrelate)

    self.params = params
    self.image_f = output

  def get_parameters(self):
    return self.params

  def get_image(self):
    return self.image_f()


class OptimizableImage(nn.Module):
  def __init__(self, w, h=None, sd=None, batch=None, channels=None):
    super(OptimizableImage, self).__init__()
    h = h or w
    batch = batch or 1
    ch = channels or 3
    sd = sd or 0.3

    shape = (batch, ch, h, w)

    self.image = nn.Parameter((torch.normal(0, sd, size=shape)))

  def get_parameters(self):
    return [self.image]

  @staticmethod
  def get_image_from_parameters(parameters):
    return torch.nn.Sigmoid()(parameters * 3)

  def get_image(self):
    # scale so that at init is roughly uniform from 0 to 1
    return self.get_image_from_parameters(self.image)




if __name__ == '__main__':
  N_steps = 1000000
  image_f = FFTImage(512, 512, 1)

  optim = torch.optim.SGD(image_f.get_parameters(), 1e-1)

  for i in range(N_steps):
    cur_image = image_f.get_image()
    loss = ((cur_image - 10)**2).mean()

    print("Loss: {}".format(loss))
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 1000 == 0:
      imshow(cur_image)
      visdom_histogram(cur_image)