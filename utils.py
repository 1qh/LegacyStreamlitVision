import os
from copy import deepcopy
from inspect import signature
from subprocess import check_output
from time import gmtime, strftime
from typing import Generator

import cv2
import numpy as np
import streamlit as st
from attrs import asdict, define
from cv2 import getOptimalNewCameraMatrix, undistort
from numpy import ndarray
from PIL import Image
from streamlit import set_page_config
from streamlit.delta_generator import DeltaGenerator
from streamlit_drawable_canvas import st_canvas
from supervision import (
  Color,
  ColorLookup,
  ColorPalette,
  Point,
  Position,
)
from vidgear.gears import VideoGear

color_dict = {
  'red': [255, 0, 0],
  'orange': [255, 100, 0],
  'yellow': [255, 200, 0],
  'green': [0, 150, 0],
  'blue': [0, 100, 255],
  'purple': [100, 0, 255],
  'black': [0, 0, 0],
  'white': [255, 255, 255],
}

base_colors = ['black', 'white']
camera_matrix = np.load('fisheye/camera_matrix.npy')
dist_coeffs = np.load('fisheye/dist_coeffs.npy')


class FisheyeRemoval:
  def __init__(self, reso: tuple[int, int] = (640, 480)):
    w, h = reso
    s = min(w, h)
    d = abs(w - h) // 2
    self.camera_matrix = camera_matrix
    self.dist_coeffs = dist_coeffs
    self.new_camera_matrix, roi = getOptimalNewCameraMatrix(
      camera_matrix, dist_coeffs, (s, s), 1, (s, s)
    )
    right, top, new_w, new_h = roi
    bottom = top + new_h
    left = right + new_w
    self.crop = slice(top, bottom), slice(right, left)
    self.slic = (
      (slice(None), slice(None))
      if w == h
      else ((slice(None), slice(d, d + h)) if w > h else (slice(d, d + w), slice(None)))
    )

  def __call__(self, f: ndarray) -> ndarray:
    return undistort(
      f[self.slic],
      self.camera_matrix,
      self.dist_coeffs,
      None,
      self.new_camera_matrix,
    )[self.crop]


class ColorClassifier:
  __slots__ = ('names', 'ycc', 'rgb')

  def __init__(self, names: list[str] = base_colors):
    d = {k: v for k, v in color_dict.items() if k in names}
    self.names = list(d.keys())

    if self.names:
      rgb_mat = np.array(list(d.values())).astype(np.uint8)
      self.ycc = rgb2ycc(rgb_mat)
      self.rgb = [tuple(map(int, i)) for i in rgb_mat]

  def closest(self, _rgb: ndarray) -> int:
    return np.argmin(
      np.sum(
        (self.ycc - rgb2ycc(_rgb[np.newaxis])) ** 2,
        axis=1,
      )
    )


@define
class Draw:
  lines: list = []
  zones: list = []

  def __str__(self) -> str:
    s = ''
    if l := len(self.lines):  # noqa: E741
      s += '\n - ' + plur(l, 'line')
    if z := len(self.zones):
      s += '\n - ' + plur(z, 'zone')
    return s

  def __len__(self) -> int:
    return len(self.lines) + len(self.zones)

  @classmethod
  def from_canvas(cls, d: list):
    return cls(
      lines=[
        (
          (i['left'] + i['x1'], i['top'] + i['y1']),
          (i['left'] + i['x2'], i['top'] + i['y2']),
        )
        for i in d
        if i['type'] == 'line'
      ],
      zones=[
        [[x[1], x[2]] for x in k]
        for k in [j[:-1] for j in [i['path'] for i in d if i['type'] == 'path']]
      ]
      + [
        [
          [i['left'], i['top']],
          [i['left'] + i['width'], i['top']],
          [i['left'] + i['width'], i['top'] + i['height']],
          [i['left'], i['top'] + i['height']],
        ]
        for i in d
        if i['type'] == 'rect'
      ],
    )


def cvt(f: ndarray) -> ndarray:
  return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def maxcam() -> tuple[int, int]:
  resos = (
    check_output(
      "v4l2-ctl -d /dev/video0 --list-formats-ext | grep x | awk '{print $NF}' | sort -u",
      shell=True,
    )
    .decode()
    .splitlines()
  )
  resos_int = [tuple(map(int, i.split('x'))) for i in resos if 'x' in i]
  prods = [np.prod(i) for i in resos_int]
  width, height = resos_int[np.argmax(prods)]

  return width, height


def plur(n: int, s: str) -> str:
  return f"{n} {s}{'s'[:n^1]}" if n else ''


def rgb2hex(rgb: tuple[int, int, int]) -> str:
  r, g, b = rgb
  return f'#{r:02x}{g:02x}{b:02x}'


def rgb2ycc(rgb: ndarray) -> ndarray:
  rgb = rgb / 255.0
  r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
  y = 0.299 * r + 0.587 * g + 0.114 * b
  cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
  cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
  return np.stack([y, cb, cr], axis=-1)


def avg_rgb(f: ndarray) -> ndarray:
  return cv2.kmeans(
    cvt(f.reshape(-1, 3).astype(np.float32)),
    1,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
    10,
    cv2.KMEANS_RANDOM_CENTERS,
  )[2][0].astype(np.int32)


def local_css(file: str):
  with open(file) as f:
    st.markdown(
      f'<style>{f.read()}</style>',
      unsafe_allow_html=True,
    )


def st_config():
  set_page_config(
    page_icon='🎥',
    page_title='ComputerVisionWebUI',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
      'Report a bug': 'https://github.com/1qh/ComputerVisionWebUI/issues/new',
    },
  )
  try:
    local_css('style.css')
  except FileNotFoundError:
    st.markdown(
      """
            <style>
            div[data-testid="stExpander"] div[role="button"] p {
              font-size: 1.5rem;
            }
            div.stButton button {
              width: 100%;
              border-radius: 20px;
            }
            div.block-container {
              padding-top: 2rem;
            }
            footer {
              visibility: hidden;
            }
            @font-face {
              font-family: "SF Pro Display";
            }
            html,
            body,
            [class*="css"] {
              font-family: "SF Pro Display";
            }
            thead tr th:first-child {
              display: none;
            }
            tbody th {
              display: none;
            }
            </style>
            """
    )


def hms(s: int) -> str:
  return strftime('%H:%M:%S', gmtime(s))


def trim_vid(path: str, begin: str, end: str) -> str:
  trim = f'trim_{path[3:]}'
  os.system(f'ffmpeg -y -i {path} -ss {begin} -to {end} -c copy {trim}')
  return trim


def filter_by_vals(d: dict, place: DeltaGenerator, text: str) -> list[int | str]:
  a = list(d.values())

  if place.toggle(text):
    return [a.index(i) for i in place.multiselect(' ', a, label_visibility='collapsed')]
  else:
    return list(d.keys())


def filter_by_keys(d: dict, place: DeltaGenerator, text: str) -> list[int | str]:
  a = list(d.keys())

  if place.toggle(text):
    return list(place.multiselect(' ', a, label_visibility='collapsed'))
  else:
    return a


def exe_button(place: DeltaGenerator, text: str, cmd: str):
  if place.button(text):
    st.code(cmd, language='bash')
    os.system(cmd)


def mycanvas(
  stroke: str,
  width: int,
  height: int,
  mode: str,
  bg: Image.Image,
  key: str,
):
  return st_canvas(
    stroke_width=2,
    fill_color='#ffffff55',
    stroke_color=stroke,
    width=width,
    height=height,
    drawing_mode=mode,
    background_image=bg,
    key=key,
  )


def canvas2draw(reso, background, is_track):
  width, height = reso
  c1, c2 = st.columns([1, 4])
  mode = c1.selectbox(
    'Draw',
    ('line', 'rect', 'polygon') if is_track else ('rect', 'polygon'),
    label_visibility='collapsed',
  )
  bg = background if c2.toggle('Background', value=True) else None
  stroke, key = ('#fff', 'e') if bg is None else ('#000', 'f')
  canvas = mycanvas(stroke, width, height, mode, bg, key)

  draw = Draw()
  if canvas.json_data is not None:
    draw = Draw.from_canvas(canvas.json_data['objects'])
    c2.markdown(draw)
  if canvas.image_data is not None and len(draw) > 0 and c1.button('Export canvas image'):
    Image.alpha_composite(
      bg.convert('RGBA'),
      Image.fromarray(canvas.image_data),
    ).save('canvas.png')
  return draw


def legacy_generator(stream: VideoGear, model) -> Generator:
  while True:
    f = stream.read()
    yield f, model(f)


def first_frame(path: str) -> Image.Image:
  stream = VideoGear(source=path).start()
  frame = Image.fromarray(cvt(stream.read()))
  stream.stop()
  return frame


def new(obj, **kw):
  expected = signature(obj).parameters.keys()
  filtered = {k: v for k, v in kw.items() if k in expected}
  return obj(**filtered)


def to_plain(ori: dict):
  d = deepcopy(ori)
  for v in d.values():
    for k2, v2 in v.items():
      match v2:
        case ColorLookup():
          v[k2] = v2.value
        case Position():
          v[k2] = v2.value
        case Color():
          v[k2] = rgb2hex(v2.as_rgb())
        case Point():
          v[k2] = v2.as_xy_int_tuple()
        case ColorClassifier():
          v[k2] = v2.names
        case ColorPalette():
          v[k2] = None
        case Draw():
          v[k2] = asdict(v2)

  for k, v in d.items():
    d[k] = {k2: v2 for k2, v2 in v.items() if v2 is not None}

  return d


def from_plain(ori: dict):
  d = deepcopy(ori)
  for v in d.values():
    for k2, v2 in v.items():
      match k2:
        case str(k2) if 'lookup' in k2:
          v[k2] = ColorLookup(v2)
        case str(k2) if 'position' in k2:
          v[k2] = Position(v2)
        case str(k2) if 'anchor' in k2:
          v[k2] = Point(v2[0], v2[1])
        case str(k2) if '_color' in k2:
          v[k2] = Color.from_hex(v2)
        case str(k2) if 'names' in k2:
          v[k2] = {int(k3): v3 for k3, v3 in v2.items()}
        case str(k2) if 'draw' in k2:
          v[k2] = Draw(**v2)

  return d


def unsnake(s: str) -> str:
  return s.replace('_', ' ').capitalize()
