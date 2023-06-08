import sys
sys.path.append('.')

import torch
from my_python_utils.common_utils import *

class MidasConstants(torch.nn.Module):
  def __init__(self, Ks):
    super(MidasConstants, self).__init__()

    assert Ks.shape[1:] == (3,3)
    bs = Ks.shape[0]

    self.d_0 = torch.nn.parameter.Parameter(torch.zeros(bs))
    self.scale = torch.nn.parameter.Parameter(torch.ones(bs))
    self.focal_factor = torch.nn.parameter.Parameter(torch.ones(bs))
    self.rx = torch.nn.parameter.Parameter(torch.zeros(bs))
    self.ry = torch.nn.parameter.Parameter(torch.zeros(bs))

    self._Ks = torch.FloatTensor(Ks)

  def get_Ks(self):
    Ks = torch.FloatTensor(self._Ks).to(self.focal_factor.device)

    Ks[:, 0, 0] *= self.focal_factor
    Ks[:, 1, 1] *= self.focal_factor

    return Ks

  def initialize_with_fit(self, pred, gt, mask0=None, robust=True, robust_loss='l1', n_robust_points=500, n_robust_fits=100):
    # replicates def recover_metric_depth
    # https://github.com/aim-uofa/AdelaiDepth/blob/de906b4ac4ad5775daedaf443fcff02a994188a0/LeReS/Train/lib/utils/evaluate_depth_error.py
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()

    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) #& (pred > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        if type(mask0).__module__ == torch.__name__:
            mask0 = mask0.cpu().numpy()
        mask0 = mask0.squeeze()
        mask0 = mask0 > 0
        mask = mask & mask0

    gt_masked = gt[mask]
    pred_masked = pred[mask]
    if robust:
      # perform ransac
      print("Performing robust fitting for midas constants with {} iterations with {} points".format(n_robust_fits,
                                                                                                     n_robust_points))
      assert len(pred_masked) > n_robust_points * 2, "Tried to initialized fit with robust formulation, " \
                                                     "but there are not enough valid points in the gt! " \
                                                     "Desired: {} available: {}".format(n_robust_points, len(pred_masked))
      errors = []
      fitted_ctts = []
      valid_fit_found = False
      for _ in tqdm(range(n_robust_fits)):
        random_is = random.sample(list(range(len(pred_masked))), n_robust_points)
        a, b = np.polyfit(pred_masked[random_is], gt_masked[random_is], deg=1)
        pred_fitted = pred_masked * a + b
        if robust_loss == 'l1':
          error = np.abs(pred_fitted - gt_masked).mean()
        if robust_loss == 'l1_80':
          diff = np.abs(pred_fitted - gt_masked)
          diff.sort()
          error = diff[:int(len(diff) * 0.8)].mean()
        if a > 0:
          valid_fit_found = True
          errors.append(error)
          fitted_ctts.append((a,b))
      if valid_fit_found:
        a, b = fitted_ctts[error.argmin()]

    else:
      a, b = np.polyfit(pred_masked, gt_masked, deg=1)
      valid_fit_found = True

    if a > 0 and valid_fit_found:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_masked)
        gt_mean = np.mean(gt_masked)
        pred_metric = pred * (gt_mean / pred_mean)

    return pred_metric

def compute_rmse(gt, pred, mask=None):
  square = (gt - pred) ** 2
  if mask is None:
    rmse_squa_sum = np.sum(square)
    rmse = np.sqrt(rmse_squa_sum)
  elif mask.sum() == 0:
    rmse =  0
  else:
    rmse_squa_sum = np.sum(square * mask)
    rmse = np.sqrt(rmse_squa_sum / mask.sum())
  return rmse

