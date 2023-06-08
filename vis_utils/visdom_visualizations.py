import plotly.graph_objs as go

import skvideo
import sys
import os

if 'anaconda' in sys.executable:
  # set ffmpeg to anaconda path
  skvideo.setFFmpegPath(os.path.split(sys.executable)[0])
else:
  skvideo.setFFmpegPath('/usr/bin')
from skvideo.io import FFmpegWriter, FFmpegReader

import tempfile

import numpy as np
import time
import imageio

import os
import warnings
import cv2

import math

from multiprocessing import Queue, Process
import datetime

import torch

PYCHARM_VISDOM='PYCHARM_RUN'

def instantiante_visdom(port, server='http://localhost'):
  return visdom.Visdom(port=port, server=server, use_incoming_socket=True, raise_exceptions=False)

if not 'NO_VISDOM' in os.environ.keys():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import visdom
    if "VISDOM_HOST" in os.environ.keys():
      visdom_host = os.environ["VISDOM_HOST"]
    else:
      print("Env variable VISDOM_HOST not found, will use localhost!")
      visdom_host = 'localhost'
    global_vis = instantiante_visdom(12890, server='http://' + visdom_host)


def list_of_lists_into_single_list(list_of_lists):
  flat_list = [item for sublist in list_of_lists for item in sublist]
  return flat_list

def visdom_heatmap(heatmap, window=None, env=None, vis=None):
  trace = go.Heatmap(z=heatmap)
  data = [trace]
  layout = go.Layout()
  fig = go.Figure(data=data, layout=layout)
  global_vis.plotlyplot(fig, win=window, env=env)

def visdom_default_window_title_and_vis(win, title, vis):
  if win is None and title is None:
    win = title = 'None'
  elif win is None:
    win = str(title)
  elif title is None:
    title = str(win)
  if vis is None:
    vis = global_vis
  return win, title, vis

def imshow_vis(im, title=None, win=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opts['title'] = title
  if im.dtype is np.uint8:
    im = im/255.0
  vis.image(im, win=win, opts=opts, env=env)

def visdom_dict(dict_to_plot, title=None, window=None, env=PYCHARM_VISDOM, vis=None, simplify_floats=True):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  vis.win_exists(title)
  if window is None:
    window = title
  dict_to_plot_sorted_keys = [k for k in dict_to_plot.keys()]
  dict_to_plot_sorted_keys.sort()
  html = '''<table style="width:100%">'''
  for k in dict_to_plot_sorted_keys:
    v = dict_to_plot[k]
    html += '<tr> <th>{}</th> <th>{}</th> </tr>'.format(k, v)
  html += '</table>'
  vis.text(html, win=window, opts=opts, env=env)

def vidshow_file_vis(videofile, title=None, window=None, env=None, vis=None, fps=10):
  # if it fails, check the ffmpeg version.
  # Depending on the ffmpeg version, sometimes it does not work properly.
  opts = dict()
  if not title is None:
    opts['title'] = title
    opts['caption'] = title
    opts['fps'] = fps
  if vis is None:
    vis = global_vis
  vis.win_exists(title)
  if window is None:
    window = title
  vis.video(videofile=videofile, win=window, opts=opts, env=env)

class MyVideoWriter():
  def __init__(self, file, fps=None, verbosity=0, *args, **kwargs):
    if not fps is None:
      kwargs['inputdict'] = {'-r': str(fps)}
    kwargs['verbosity'] = verbosity
    assert verbosity in range(2), "Verbosity should be between 0 or 1"
    self.video_writer = FFmpegWriter(file, *args, **kwargs)

  def writeFrame(self, im):
    if len(im.shape) == 3 and im.shape[0] == 3:
      transformed_image = im.transpose((1,2,0))
    elif len(im.shape) == 2:
      transformed_image = np.concatenate((im[:,:,None], im[:,:,None], im[:,:,None]), axis=-1)
    else:
      transformed_image = im
    self.video_writer.writeFrame(transformed_image)

  def close(self):
    self.video_writer.close()

class MyVideoReader():
  def __init__(self, video_file):
    if video_file.endswith('.m4v'):
      self.vid = imageio.get_reader(video_file, format='.mp4')
    else:
      self.vid = imageio.get_reader(video_file)
    self.frame_i = 0

  def get_next_frame(self):
    try:
      return np.array(self.vid.get_next_data().transpose((2,0,1)))
    except:
      return None

  def get_n_frames(self):
    return int(math.floor(self.get_duration_seconds() * self.get_fps()))

  def get_duration_seconds(self):
    return self.vid._meta['duration']

  def get_fps(self):
    return self.vid._meta['fps']

  def position_cursor_frame(self, i):
    assert i < self.get_n_frames()
    self.frame_i = i
    self.vid.set_image_index(self.frame_i)

  def get_frame_i(self, i):
    old_frame_i = self.frame_i
    self.position_cursor_frame(i)
    frame = self.get_next_frame()
    self.position_cursor_frame(old_frame_i)
    return frame

  def is_opened(self):
    return not self.vid.closed

  # encoded as apple ProRes mov
# ffmpeg -i input.avi -c:v prores_ks -profile:v 3 -c:a pcm_s16le output.mov
# https://video.stackexchange.com/questions/14712/how-to-encode-apple-prores-on-windows-or-linux
def get_video_writer(videofile, fps=10, verbosity=0, extension='mov'):
  extension = extension.replace('.', '')
  writer = MyVideoWriter(videofile + '.' + extension, verbosity=verbosity, inputdict={'-r': str(fps)},
                                                                  outputdict={
                                                                    '-c:v': 'prores_ks',
                                                                    '-profile:v': '3',
                                                                    '-c:a': 'pcm_s16le'})
  return writer


def vidshow_gif_path(gif_path, title=None, win=None, env=None, vis=None):
  # as html base 64, as gif display is not implemented https://github.com/fossasia/visdom/issues/685
  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)
  opts = dict()
  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opts['title'] = title
  html = '''
    <table>
    <tr>
      <th>Month</th>
      <th>Savings</th>
    </tr>
    <tr>
      <td>January</td>
      <td>$100</td>
    </tr>
  </table>
  '''

  import base64
  with open(gif_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
  html_2= '''
  <img src='data:image/gif;base64,R0lGODlhyADIAPIFAP%2FyAAoKCgAAAcRiAO0cJAAAAAAAAAAAACH%2FC05FVFNDQVBFMi4wAwEAAAAh%2BQQJCgAFACwAAAAAyADIAAAD%2F1i63P4wykmrvTjrzbv%2FYCiOZGmeaKqubOu%2BcCzPdG3feK7vfO%2F%2FwKBwSCwaj8ikcslsOp%2FQqHRKrVqv2Kx2y%2B16v%2BCweEwum8%2FotHrNbrvf8Lh8Tq%2Fb7%2Fi8fs%2Fv%2B%2F%2BAgYKDhIWGh4iJiouMjY6PkJGSk5SVlpeYmZqbnJ2en6ChoqOkpaanqKmqq6ytrq%2BwsbKztLW2t7i5uru8vb6%2FwMHCw8TFxsfIycrLzM3Oz9DR0tPU1dbX2Nna29zd3t%2Fg4eLj5OXmkwLpAszq68vt7OrI7QH1AfDD9Pb4wvr1%2FMH83ZP3S6C9gemAGdxH0NfCfw17PUTozqG6gwcBkph4EP%2FSRI0jONrzeBEjxIQnRNYjmc7kyYolVAZgKcAlRRDt2gHYybPnzo6PPkbkkFOdz6MAgDoSitJD0XRIfSptxBQm0adRe05lpBMpSAtds2bduiisz68VzIo9SlaRWp5oKbxdy7NtorkA4k7AS9cumKeAA9vMiNXr0L1G6a71%2ByWw45yDGRaNqtcBX8U%2FR555zLlmZIp4Kze4jJmxl86PP4NOfFQ0A9KKTWeReRCzbcNNI8C%2BLRsLbXu3g8M9bJm1cKS9r%2Fyudzy46N22k1tZHqD57efGrdfVbEamVuDazxIvAF249NklI39nHr4n2vLBz%2FtOP3h99fbDc7%2FOjj%2Fzys3%2F9NlkX387vcdff%2FJtgVpL4PVnIFTHqQbHgp6x5%2BB48Nln04QL1kbggwI0J%2BEbFHp4oX4LZLhdZICYiJ9sZg0g4wD2MeJiezAaNyONK860yI3h5QjhTjvW%2BGODL3Knm44zGqmIi6EdmJSSELSz45UzJqgHlFLiJaQAWGKpZR5cDimemU4umU6YV46JR5kh4hYnW1Q%2BYCWbWdZpyEEE9EnAbX7%2B2SOFd4qpZyF8%2BgmoooMSumaYbt6RaJ%2BLUtqoo2xGasekgmIWqH2OPmrof44AqV2RPKEqlqZ9mGqdqgDAGhWrfLjaHKyyIneojUi2h2uTi%2B36iGq3%2FSpjX8KW%2Blmxh8AS2exYyTZCrG3G8rhqtLyqR%2B2zudJJaie2EpgmJ%2BGK65%2BPnpRrLq2HqCsuu3v2aq636IIr77zjbuIugfAiei%2B%2B54LiooA9DuxSvpoYbJKGSBIc8CcKY8SwhVMu3KPADR9ccMYWPyyKXSAf6pq%2Bh4b87X4oflzyyienOB7GLStgcr0oW%2FVEAgAh%2BQQJCgAFACwsAHwAbABMAAAD%2F1i63P4wPkGFvDjrzXO1XSiOJPSVaKpK5%2Bq%2B4RfMQQvfOCPTdu6%2Fu1nvR0QFa5WiUnSkISnL6KbJS0qvrIrTOcR6FVSh9UsuhJ%2Bg29n5PXdXa1pbuxVDcfHZnFK3p2F5AXsCfWgpHx8AiouMimxebmMkiBWNlgCPWJF3JZQUl42ZV5t%2FI54CoIyiUomXbx6VqbKrUa2Wrxi2spe0S7qMuBe%2Fu6pykLG3khzDxI7GYKfRlIVcnqDBDszNxXoL0t901Gja2A3a287d0ODS4n7kysLI6Jai7N%2Fu4%2FPA8Vmf9Lyq8MlHA6BBAOXOHaw2kGCAgwAT7oO4iCEhhw8pbpP4T%2F8jNzQYM3rcxRHVyIrPzISj9vHkolcKNdpbWailS4T9VHa8mU6QN5p9bLqEOdHlzIYsUc7gSXQnz1462TlhmjNmqny57l1cerOpSYNY5d2b2rVq0WZh%2FUktWJaTubPE0qogazSliXkD8g74KIXuSag68OrlG8XvSMA%2Fd%2Brdq9TnEsMeEa%2F7CmAx4cdsFcFz2jgrhcWg9UqG4Xcz5csRPoQOPfpF6bPaRqtevbi1i9ecNZ%2BVXYF2bbtEnBAYToAe8eKNtSKibXuFcOLGoSdX3nt187k0jkcf%2FpF6ddbAfzznjk77dO%2FMwyuBrNHyIvez1PfNfBJ%2B5cG7rudgT9G%2BfVCl%2BuHAH0T%2B4RefOmUskA89BeYVl3xeLIhOg4wd6FiCCki4DYUPIoihhs1wmB%2BEGGZIH08AkljigCj2VOIFLLYYIBYxojjjFTU%2BpeKHJ7YYyo4J5njTjfNx5WNAHr7YgF81NcZkUJ0pCcGTdXxE5RaoScnAlVzS16SLWjrQpZGYQNnTlWFKANWa6pWTZgFsJmminFG9iUGcF27ZZk52Kqgenne5NUICACH5BAUKAAUALDAAfABsAEwAAAP%2FWLrc%2FjA%2BQYW8OOvNc7VdKI4k9JVoqkrn6r7hF8xBC984I9N27r%2B7We9HRAVrlaJSdKQhKcvopslLSq%2BsitM5xHoVVKH1Sy6En6Db2fk9d1drWlu7FUNx8dmcUrenYXkBewJ9aCkfHwCKi4yKbF5uYySIFY2WAI9YkXcllBSXjZlXm38jngKgjKJSiZdvHpWpsqtRrZavGLayl7RLuoy4F7%2B7qnKQsbeSHMPEjsZgp9GUhVyeoMEOzM3FegvS33TUaNrYDdrbzt3Q4NLifuTKwsjolqLs3%2B7j88DxWZ%2F0vKrwyUcDoEEA5c4drDaQYICDABPug7iIISGHDyluk%2FhP%2FyM3NBgzetzFEdXIis%2FMhKP28eSiVwo12ltZqKVLhP1UdryZTpA3mn1suoQ50eXMhixRzuBJdCfPXjrZOWGaM2aqfLnuXVx6s6lJg1jl3ZvatWrRZmH9SS1YlpO5s8TSqiBrNKWJeQPyDvgohe5JqDrw6uUbxe9IwD936t2r1OcSwx4Rr%2FsKYDHhx2wVwXPaOCuFxaD1SobhdzPlyxE%2BhA49%2BkXps9pGq169uLWL15w1n5VdgXZtu0ScEBhOgB7x4o21IqJte4Vw4sahJ1fee3XzuTSORx%2F%2BkXp11sB%2FPOeOTvt078zDK4Gs0fIi97PU9818En7lwbuu52BP0b59UKX64cAfRP7hF586ZSyQDz0F5hWXfF4siE6DjB3oWIIKSLgNhQ8iiKGGzXCYH4QYZkgfTwCSWOKAKPZU4gUsthggFjGiOOMVNT6l4ocnthjKjgnmeNON83HlY0AevtiAXzU1xmRQnSkJwZN1fETlFqhJycCVXNLXpItaOtClkZhA2dOVYUoA1ZrqlZNmAWwmaaKcUb2JQZwXbtlmTnYqqB6ed7k1QgIAOw%3D%3D' />
  '''
  html_2 = '''<img src='data:image/gif;base64,{}' />  '''.format(str(encoded_string)[2:-1])
  vis.text(html_2, win=win, opts=opts, env=env)

  return gif_path

def vidshow_vis(frames, title=None, window=None, env=None, vis=None, biggest_dim=None, fps=10, verbosity=0):
  # if it does not work, change the ffmpeg. It was failing using anaconda ffmpeg default video settings,
  # and was switched to the machine ffmpeg.
  if vis is None:
    vis = global_vis
  if frames.shape[1] == 1 or frames.shape[1] == 3:
    frames = frames.transpose(0, 2, 3, 1)
  if frames.shape[-1] == 1:
    #if one channel, replicate it
    frames = np.tile(frames, (1, 1, 1, 3))
  if not frames.dtype is np.uint8:
    frames = np.array(frames * 255, dtype='uint8')
  # visdom available extensions/mimetypes
  # mimetypes (audio) = {'wav': 'wav', 'mp3': 'mp3', 'ogg': 'ogg', 'flac': 'flac'}
  # mimetypes (video) = {'mp4': 'mp4', 'ogv': 'ogg', 'avi': 'avi', 'webm': 'webm'}
  videofile = '/tmp/%s.webm' % next(tempfile._get_candidate_names())
  writer = MyVideoWriter(videofile, inputdict={'-r': str(fps)}, verbosity=verbosity)
  for i in range(frames.shape[0]):
    if biggest_dim is None:
      actual_frame = frames[i]
    else:
      actual_frame = np.array(np.transpose(scale_image_biggest_dim(np.transpose(frames[i]), biggest_dim)), dtype='uint8')
    try:
      writer.writeFrame(actual_frame)
    except Exception as e:
      print(e)
      print("If this fails, copy paste the ffmpeg command, by going into skvideo.io.ffmpeg.FFmpegWriter._createProcess"
            "as probably there are system libraries that are not been properly installed, which can be installed through conda.")
      print("e.g. libopenh264.so.5: cannot open shared object file: No such file or directory")
      print("Changing to .webm from .mp4 also helped one day that it was not working with a new conda install.")
  writer.close()

  os.chmod(videofile, 0o777)
  vidshow_file_vis(videofile, title=title, window=window, env=env, vis=vis, fps=fps)
  return videofile

def scale_image_biggest_dim(im, biggest_dim):
  #if it is a video, resize inside the video
  if im.shape[1] > im.shape[2]:
    scale = im.shape[1] / (biggest_dim + 0.0)
  else:
    scale = im.shape[2] / (biggest_dim + 0.0)
  target_imshape = (int(im.shape[1]/scale), int(im.shape[2]/scale))
  if im.shape[0] == 1:
    im = myimresize(im[0], target_shape=(target_imshape))[None,:,:]
  else:
    im = myimresize(im, target_shape=target_imshape)
  return im

def myimresize(img, target_shape, interpolation_mode=cv2.INTER_NEAREST):
  max = img.max(); min = img.min()
  uint_mode = img.dtype == 'uint8'

  assert len(target_shape) == 2, "Passed shape {}. Should only be (height, width)".format(target_shape)
  if max > min and not uint_mode:
    # normalize image and undo normalization after the resize
    img = (img - min)/(max - min)
  if len(img.shape) == 3 and img.shape[0] in [1,3]:
    if img.shape[0] == 3:
      img = np.transpose(cv2.resize(np.transpose(img, (1,2,0)), target_shape[::-1], interpolation=interpolation_mode), (2,0,1))
    else:
      img = cv2.resize(img[0], target_shape[::-1], interpolation=interpolation_mode)[None,:,:]
  else:
    img = cv2.resize(img, target_shape[::-1], interpolation=interpolation_mode)
  if max > min and not uint_mode:
    # undo normalization
    return (img*(max - min) + min)
  else:
    return img

class ThreadedVisdomPlotter():
  # plot func receives a dict and gets what it needs to plot
  def __init__(self, plot_func, use_threading=True, queue_size=10, force_except=False):
    self.queue = Queue(queue_size)
    self.plot_func = plot_func
    self.use_threading = use_threading
    self.force_except = force_except
    def plot_results_process(queue, plot_func):
        # to avoid wasting time making videos
        while True:
            try:
                if queue.empty():
                    time.sleep(1)
                    if queue.full():
                        print("Plotting queue is full!")
                else:
                    actual_plot_dict = queue.get()
                    env = actual_plot_dict['env']
                    time_put_on_queue = actual_plot_dict.pop('time_put_on_queue')
                    visdom_dict({"queue_put_time": time_put_on_queue}, title=time_put_on_queue, window='params', env=env)
                    print("Plotting...")
                    plot_func(**actual_plot_dict)
                    continue
            except Exception as e:
                if self.force_except:
                  raise e
                print('Plotting failed wiht exception: ')
                print(e)
    if self.use_threading:
      Process(target=plot_results_process, args=[self.queue, self.plot_func]).start()

  def _detach_tensor(self, tensor):
    if tensor.is_cuda:
      tensor = tensor.detach().cpu()
    tensor = np.array(tensor.detach())
    return tensor

  def _detach_dict_or_list_torch(self, list_or_dict):
    # We put things to cpu here to avoid er
    if type(list_or_dict) is dict:
      to_iter = list(list_or_dict.keys())
    elif type(list_or_dict) is list:
      to_iter = list(range(len(list_or_dict)))
    else:
      return list_or_dict
    for k in to_iter:
      if type(list_or_dict[k]) is torch.Tensor:
        list_or_dict[k] = self._detach_tensor(list_or_dict[k])
      else:
        list_or_dict[k] = self._detach_dict_or_list_torch(list_or_dict[k])
    return list_or_dict

  def clear_queue(self):
    while not self.queue.empty():
      self.queue.get()

  def is_queue_full(self):
    if not self.use_threading:
      return False
    else:
      return self.queue.full()

  def n_queue_elements(self):
      if not self.use_threading:
        return 0
      else:
        return self.queue.qsize()

  def put_plot_dict(self, plot_dict):
    try:
      assert type(plot_dict) is dict
      assert 'env' in plot_dict, 'Env to plot not found in plot_dict!'
      plot_dict = self._detach_dict_or_list_torch(plot_dict)
      if self.use_threading:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        plot_dict['time_put_on_queue'] = timestamp
        self.queue.put(plot_dict)
      else:
        self.plot_func(**plot_dict)
    except Exception as e:
      if self.force_except:
        raise e
      print('Putting onto plot queue failed with exception:')
      print(e)



if __name__ == '__main__':
  def plot_func(env):
    time.sleep(1)
    #raise Exception("Test exception!")
  a = ThreadedVisdomPlotter(plot_func,  use_threading=True, queue_size=10, force_except=False)
  for k in range(20):
    a.put_plot_dict({'env': 'env'})
    if a.is_queue_full():
      a.clear_queue()
    print(a.queue.qsize())

  heatmap = [[1, 20, 30],
   [20, 1, 60],
   [30, 60, 1]]
  heatmap = np.random.normal(scale=1, size=(36,10))
  visdom_heatmap(np.array(heatmap))


