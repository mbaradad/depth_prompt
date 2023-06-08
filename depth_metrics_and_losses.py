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


def test_correct_shapes(gt, pred, mask):
  assert len(gt.shape) == 3 and gt.shape == pred.shape and (mask is None or gt.shape == mask.shape)

def compute_rmse(gt, pred_original, mask=None, align_scale=False, align_shift=False, return_per_sample=False):
  test_correct_shapes(gt, pred_original, mask)

  if align_scale and align_shift:
    pred, scale_shift_batch = recover_scale_shift_depth(pred_original[:,None], gt[:,None], mask[:,None])
    pred = pred[:,0]
  elif align_scale:
    pred, scale_batch = recover_scale_depth(pred_original[:, None], gt[:, None], mask[:, None])
    pred = pred[:,0]
  elif align_shift:
    raise Exception("Not implemented only shift!")
  else:
    pred = pred_original

  square = (gt - pred) ** 2
  if mask is None:
    rmse_squa_sum = torch.mean(square)
    rmse = torch.sqrt(rmse_squa_sum + 1e-12)
  else:
    rmse_squa_sum = (square * mask).sum(-1).sum(-1)
    rmse = torch.sqrt(rmse_squa_sum / (mask.sum(-1).sum(-1) + 1e-6) + 1e-12)

  if return_per_sample:
    return rmse
  else:
    return rmse.mean()

def si_rmse(gt, pred, mask, return_per_sample=False):
  return compute_rmse(gt, pred, mask, align_scale=True, return_per_sample=return_per_sample)

def rmse(gt, pred, mask):
  return compute_rmse(gt, pred, mask, align_scale=False)

def l1(gt, pred, mask):
  test_correct_shapes(gt, pred, mask)

  square = torch.abs(gt - pred)
  if mask is None:
    l1 = torch.mean(square)
  else:
    l1_sum = (square * mask).sum(-1).sum(-1)
    l1 = l1_sum / (mask.sum(-1).sum(-1) + 1e-6)

  return l1.mean()

def compute_cos_sim(gt_normals, pred_normals, mask):
  assert len(gt_normals.shape) == 4 and gt_normals.shape == pred_normals.shape and len(mask.shape) == 3 and  \
    (mask is None or (gt_normals.shape[0] == mask.shape[0] and gt_normals.shape[2:] == mask.shape[1:]))

  if not mask is None:
    normals_cos_sim = ((F.cosine_similarity(pred_normals, gt_normals, dim=1)) * mask).sum(-1).sum(-1) / (mask.sum(-1).sum(-1) + 1e-6)
  else:
    normals_cos_sim = ((F.cosine_similarity(pred_normals, gt_normals, dim=1))).mean(-1).mean(-1)

  return normals_cos_sim

def cos_sim_loss(gt, pred, mask):
  return (0.5 - 0.5 * compute_cos_sim(gt, pred, mask)).mean()

def recover_scale_depth(pred, gt, gt_mask, min_threshold=1e-8, max_threshold=1e8):
  assert len(pred.shape) == len(gt.shape) == len(gt_mask.shape) == 4, "Tensors should have dim 4"
  b, c, h, w = pred.shape
  mask = (gt > min_threshold) & (gt < max_threshold)  # [b, c, h, w]
  if not gt_mask is None:
    mask = mask * gt_mask

  scales = (pred * gt * mask).sum(-1).sum(-1).sum(-1) / ((pred ** 2 * mask).sum(-1).sum(-1).sum(-1) + 1e-6)
  return pred * scales[:,None,None,None], scales

def recover_scale_shift_depth(pred, gt, gt_mask, min_threshold=1e-8, max_threshold=1e8):
  b, c, h, w = pred.shape
  mask = (gt > min_threshold) & (gt < max_threshold)  # [b, c, h, w]
  if not gt_mask is None:
    mask = mask & gt_mask.bool()

  EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
  scale_shift_batch = []
  ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)
  for i in range(b):
    mask_i = mask[i, ...]
    pred_valid_i = pred[i, ...][mask_i].detach()
    ones_i = ones_img[mask_i]
    pred_valid_ones_i = torch.stack((pred_valid_i, ones_i), dim=0)  # [c+1, n]
    A_i = torch.matmul(pred_valid_ones_i, pred_valid_ones_i.permute(1, 0))  # [2, 2]
    A_inverse = torch.inverse(A_i + EPS)

    gt_i = gt[i, ...][mask_i]
    B_i = torch.matmul(pred_valid_ones_i, gt_i)[:, None]  # [2, 1]
    scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]
    scale_shift_batch.append(scale_shift_i)
  scale_shift_batch = torch.stack(scale_shift_batch, dim=0)  # [b, 2, 1]
  ones = torch.ones_like(pred)
  pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
  pred_scaled_shifted = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2),
                                  scale_shift_batch)  # [b, h*w, 1]
  pred_scaled_shifted = pred_scaled_shifted.permute(0, 2, 1).reshape((b, c, h, w))
  return pred_scaled_shifted, scale_shift_batch

def normalize_scale_depth(depth, mask):
  return depth / (depth[mask == 1].mean() + 1e-10) * 5

# find
def object_loss():
  return "Todo"

if __name__ == '__main__':
  data = np.load('/data/vision/torralba/movies_sfm/projects/normals_acc/datasets/dumped_datasets/google_scans_dumped/Schleich_Allosaurus.npz')

  img = totorch(data['img'])
  gt_depth = totorch(data['depth'])
  mask = totorch(data['depth_mask'])

  depth = gt_depth
  print(compute_rmse(gt_depth[None], depth[None], mask[None]))

  scale = 300000.14
  shift = 0

  depth = scale * (gt_depth + shift)

  print("RMSE with alignment: {}".format(compute_rmse(gt_depth[None], depth[None], mask[None], align_scale=True, align_shift=False)))
  print("RMSE without alignment: {}".format(compute_rmse(gt_depth[None], depth[None], mask[None], align_scale=False, align_shift=False)))
