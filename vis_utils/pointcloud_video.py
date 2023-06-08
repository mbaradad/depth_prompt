from my_python_utils.common_utils import *

from my_python_utils.vis_utils.simple_3dviz.simple_3dviz.renderables import Spherecloud
from my_python_utils.vis_utils.simple_3dviz.simple_3dviz.utils import render
from my_python_utils.vis_utils.simple_3dviz.simple_3dviz import Mesh
from my_python_utils.vis_utils.simple_3dviz.simple_3dviz.behaviours.io import StoreFramesAsList
from my_python_utils.vis_utils.simple_3dviz.simple_3dviz.behaviours.movements import CameraTrajectory
from my_python_utils.vis_utils.simple_3dviz.simple_3dviz.behaviours.trajectory import Circle
from my_python_utils.vis_utils.simple_3dviz.simple_3dviz import Lines

def render_pointcloud(pcl, colors, K=None, valid_mask=None, debug=False, add_camera_frustrum=False, up_and_down=False):
  if add_camera_frustrum:
    assert not K is None, "K should be passed when add_camera_frustrum is set to True"
    w, h = K[:2, 2] * 2
    x_0_y_0 = np.linalg.inv(K) @ np.array((0,0,1))
    x_0_y_1 = np.linalg.inv(K) @ np.array((0,h,1))
    x_1_y_0 = np.linalg.inv(K) @ np.array((w,0,1))
    x_1_y_1 = np.linalg.inv(K) @ np.array((w,h,1))

    x_0_y_0 /= np.linalg.norm(x_0_y_0) / 0.3
    x_0_y_1 /= np.linalg.norm(x_0_y_1) / 0.3
    x_1_y_0 /= np.linalg.norm(x_1_y_0) / 0.3
    x_1_y_1 /= np.linalg.norm(x_1_y_1) / 0.3

    l = Lines([
      [0.0, 0.0, 0.0],
      x_0_y_0.tolist(),
      [0.0, 0.0, 0.0],
      x_0_y_1.tolist(),
      [0.0, 0.0, 0.0],
      x_1_y_0.tolist(),
      [0.0, 0.0, 0.0],
      x_1_y_1.tolist(),

      x_0_y_0.tolist(),
      x_1_y_0.tolist(),
      x_1_y_0.tolist(),
      x_1_y_1.tolist(),
      x_1_y_1.tolist(),
      x_0_y_1.tolist(),
      x_0_y_1.tolist(),
      x_0_y_0.tolist(),

    ],
      colors=np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],

        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
      ]), width=0.01)

  assert pcl.shape[0] == colors.shape[0] == 3
  assert pcl.shape[1:] == colors.shape[1:]

  assert valid_mask is None or valid_mask.shape == colors.shape[1:]

  centers = np.array(pcl.reshape(3, -1)).transpose()
  pcl_colors = np.array(colors.reshape(3, -1)).transpose()

  if not valid_mask is None:
    valid_mask = valid_mask.reshape(-1)
    centers = centers[valid_mask == 1, :]
    pcl_colors = pcl_colors[valid_mask == 1, :]

  # scale
  z_at_80 = np.quantile(centers[:, 2], 0.80)
  centers /= z_at_80

  sizes = np.zeros(len(centers)) + 0.002

  s = Spherecloud(centers=centers, sizes=sizes, colors=pcl_colors)

  #m = Mesh.from_file("models/baby_yoda.stl", color=(0.1, 0.8, 0.1))
  #m.to_unit_cube()
  init_camera_pos = (1, 0, 0)
  camera_target = (0, 0, 1)
  objects_to_render = [s]
  if debug:
    sphere_center = Spherecloud(centers=np.array(camera_target)[None,:], sizes=0.1, colors=pcl_colors[0:1])
    objects_to_render.append(sphere_center)
  if add_camera_frustrum:
    objects_to_render.append(l)

  frame_list_storer = StoreFramesAsList()
  render(objects_to_render,
         n_frames=60,
         camera_position=init_camera_pos,
         camera_target=camera_target,
         up_vector=(0, -1, 0),
         size=(256, 256),
         light=(0,0,0),
         behaviours=[CameraTrajectory(Circle(center=camera_target,
                                             point=init_camera_pos,
                                             normal=[0, -1, 0]), speed=0.004),
                     frame_list_storer])
  frames = [k[:,:,:3].transpose((2,0,1)) for k in frame_list_storer.get_frames()]
  frames = list_of_lists_into_single_list([frames[30:], frames[::-1], frames[0:30]])
  if debug:
    imshow(frames)

  return frames

if __name__ == '__main__':
  toby_file = '/data/vision/torralba/movies_sfm/projects/normals_acc/datasets/nerfies_outputs/toby-sit/0050.npz'
  toby_result = np.load(toby_file)

  depth = toby_result['depth']
  image = toby_result['rgb']

  _, h, w = image.shape
  f = w

  K = np.array(((f, 0, w / 2),
                (9, f, h / 2),
                (0, 0,     1)))

  # depth = depth * 0 + 1
  pcl = pixel2cam(totorch(depth)[None], totorch(K)[None])[0]

  frames = render_pointcloud(pcl, image, debug=True, add_camera_frustrum=True, K=K)
  imshow(frames[0], title='initial_frame')

  imshow(frames[1:] + frames[::-1][:-1], gif=False, title='dog_gif')