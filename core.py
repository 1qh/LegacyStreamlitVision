import json
from copy import deepcopy
from inspect import signature
from pathlib import Path
from typing import Generator

import numpy as np
import streamlit as st
from attrs import asdict
from numpy import ndarray
from PIL import Image
from streamlit import sidebar as sb
from supervision import (
  BlurAnnotator,
  BoundingBoxAnnotator,
  BoxAnnotator,
  BoxCornerAnnotator,
  CircleAnnotator,
  ColorAnnotator,
  ColorLookup,
  Detections,
  DotAnnotator,
  EllipseAnnotator,
  HaloAnnotator,
  HeatMapAnnotator,
  LabelAnnotator,
  MaskAnnotator,
  PixelateAnnotator,
  PolygonAnnotator,
  Position,
  TraceAnnotator,
  TriangleAnnotator,
  VideoInfo,
)

from custom_annotator import (
  AreaAnnotator,
  ColorClassifier,
  ColorClassifierAnnotator,
  CountAnnotator,
  FpsAnnotator,
  LineAndZoneAnnotator,
)
from model import Model, ModelInfo
from utils import (
  FisheyeFlatten,
  canvas2draw,
  color_dict,
  exe_button,
  first_frame,
  from_plain,
  maxcam,
  rgb2hex,
  to_plain,
  unsnake,
)

all_anns = {
  AreaAnnotator,
  BlurAnnotator,
  BoundingBoxAnnotator,
  BoxAnnotator,
  BoxCornerAnnotator,
  CircleAnnotator,
  ColorAnnotator,
  ColorClassifierAnnotator,
  CountAnnotator,
  DotAnnotator,
  EllipseAnnotator,
  FpsAnnotator,
  HaloAnnotator,
  HeatMapAnnotator,
  LabelAnnotator,
  LineAndZoneAnnotator,
  MaskAnnotator,
  PixelateAnnotator,
  PolygonAnnotator,
  TraceAnnotator,
  TriangleAnnotator,
}


all_class = {i.__name__[:-9]: i for i in all_anns}
all_names = list(all_class.keys())
all_default = {}

for i in all_names:
  sig = {}
  for j in signature(all_class[i]).parameters.items():
    sig |= {j[0]: j[1].default}
  all_default[i] = sig

custom_defaults = {
  'text_padding': 1,
  'text_thickness': 1,
  'thickness': 1,
}

for v in all_default.values():
  for k, d in custom_defaults.items():
    if k in v:
      v[k] = d


all_plain = to_plain(all_default)


class Annotator:
  def __init__(self, model: Model, anns: dict = None):
    if anns is None:
      anns = {}
    self.unneeded = model.task in ('classify', 'pose')
    self.model = model
    self.names = self.model.names

    if 'Label' in anns:
      self.label = anns['Label']
      del anns['Label']
    else:
      self.label = None

    if 'Trace' in anns:
      self.trace = anns['Trace']
      del anns['Trace']
    else:
      self.trace = None

    self.linezone = None
    if 'LineAndZone' in anns:
      self.linezone: LineAndZoneAnnotator = anns['LineAndZone']

    self.anns = anns

  @classmethod
  def load(cls, path: str):
    d = json.load(open(path))
    model = Model(ModelInfo(**d['model']))
    config = from_plain(d['config'])
    preprocessors = d['preprocessors']
    if 'FisheyeFlatten' in preprocessors:
      model.preprocessors.append(FisheyeFlatten(d['reso']))
    anns = {i: all_class[i](**config[i]) for i in config}
    return cls(model, anns)

  def __call__(
    self,
    f: ndarray,
    det: Detections,
  ) -> ndarray:  # sourcery skip: low-code-quality
    names = self.names
    if self.label:
      f = self.label.annotate(
        f,
        det,
        labels=[
          f'{conf:0.2f} {names[cl] if len(names) else cl}' + (f' {track_id}' if track_id else '')
          for _, _, conf, cl, track_id in det
        ],
      )
    if self.trace:
      try:
        f = self.trace.annotate(f, det)
      except Exception as e:
        print(e)

    for v in self.anns.values():
      f = v.annotate(f, det)
    return f

  def gen(self, source: str | int) -> Generator:
    for f, (det, fallback) in self.model(source):
      yield self(f, det), fallback

  def from_frame(self, f: ndarray) -> tuple[ndarray, ndarray]:
    det, fallback = self.model.from_frame(f)
    return self(f, det), fallback

  @classmethod
  def ui(cls, source: str | int):  # sourcery skip: low-code-quality
    if source:
      model = Model.ui()
      reso = VideoInfo.from_video_path(source).resolution_wh
      background = first_frame(source)
    else:
      model = Model.ui(track=False)
      ex = sb.expander('For camera', expanded=True)
      reso = maxcam()
      if ex.toggle('Custom resolution'):
        c1, c2 = ex.columns(2)
        reso = (
          c1.number_input('Width', 1, 7680, 640, 1),
          c2.number_input('Height', 1, 4320, 480, 1),
        )
      background = None
      if ex.toggle('Annotate from image'):
        if ex.toggle('Upload'):
          background = ex.file_uploader(' ', label_visibility='collapsed', key='u')
        if ex.toggle('Shoot'):
          background = st.camera_input('Shoot')
      if background:
        model.predict_image(background)
        background = Image.open(background).resize(reso)
      ex.write('**Notes:** Track & line counts only work on native run')

    preprocessors = []
    ex0 = sb.expander('Experimental Features')
    if ex0.toggle('Fisheye Flatten'):
      aspect_ratio = 1
      if ex0.toggle('Custom aspect ratio'):
        c1, c2 = ex0.columns(2)
        w = c1.number_input('Width', 1, 16, 16, 1)
        h = c2.number_input('Height', 1, 16, 9, 1)
        aspect_ratio = w / h

      flattener = FisheyeFlatten(reso, aspect_ratio)
      model.preprocessors.append(flattener)
      preprocessors.append('FisheyeFlatten')
      background = Image.fromarray(flattener(np.array(background)))

    names = model.names
    task = model.task
    is_track = model.tracker is not None
    is_det = task == 'detect'
    is_seg = task == 'segment'

    if task in ('pose', 'classify'):
      return cls(model, None)

    draw = canvas2draw(reso, background, is_track)

    base_anns: set = {'Fps', 'Label', 'Count'}

    if is_det:
      base_anns.add('BoxCorner')
    if is_seg:
      base_anns.add('Halo')
    if is_track:
      base_anns.add('Trace')
    if len(draw):
      base_anns.add('LineAndZone')

    ann_names = sb.multiselect('Annotators', all_names, base_anns)

    # config_plain = all_plain
    origin_config_plain = {k: v for k, v in all_plain.items() if k in ann_names}

    config_plain = deepcopy(origin_config_plain)

    for k, v in config_plain.items():
      ex = sb.expander(k, expanded=True)

      ini_conf = {}
      for k2, v2 in v.items():
        key = f'{k}_{k2}'
        ini_conf[key] = list(v2) if isinstance(v2, tuple) else v2

        tit = unsnake(k2)
        match k2:
          case str(k2) if 'lookup' in k2:
            lookup_list = ColorLookup.list()
            v[k2] = ex.selectbox(
              tit,
              lookup_list,
              lookup_list.index(v2),
              key=key,
            )

          case str(k2) if 'anchor' in k2:
            ex.subheader('Position')
            v[k2] = list(v[k2])
            c1, c2 = ex.columns(2)
            v[k2][0] = c1.slider('x', 0, reso[0], v2[0], 1, key=f'{key}_x')
            v[k2][1] = c2.slider('y', 0, reso[1], v2[1], 1, key=f'{key}_y')

          case str(k2) if 'position' in k2:
            pos_list = Position.list()
            v[k2] = ex.selectbox(
              tit,
              pos_list,
              pos_list.index(v2),
              key=key,
            )

          case str(k2) if '_color' in k2:
            v[k2] = ex.color_picker(tit, v2, key=key)

        match v2:
          case bool():
            v[k2] = ex.toggle(tit, v2, key=key)
          case int():
            abso = abs(v2)
            min_val = min([0, v2, 10 * v2 + 1])
            max_val = max([0, abso, 10 * abso + 1])
            v[k2] = ex.number_input(tit, min_val, max_val, v2, 1, key=key)
          case float():
            v[k2] = ex.number_input(tit, 0.0, 10 * v2 + 1.0, v2, 0.1, key=key)
          case dict():
            pass

      cur_conf = {f'{k}_{ki}': va for ki, va in v.items()}

      if cur_conf != ini_conf:
        diff = {k: v for k, v in ini_conf.items() if v != cur_conf[k]}
        for kd, vd in diff.items():
          ex.write(f'Default {unsnake(kd.removeprefix(k))} = {vd}')
        ex.button(
          'Reset',
          key=k,
          disabled=True,
          help='Resetting to default values is not implemented',
        )

      match k:
        case 'ColorClassifier':
          clf = ColorClassifier()
          ex.subheader('Classes')
          all_colors = color_dict.keys()
          color_names = (
            ex.multiselect(' ', all_colors, ['black', 'white']) if ex.toggle('Custom colors') else all_colors
          )
          if len(color_names) > 0:
            clf = ColorClassifier(color_names)
            for c, rgb in zip(clf.names, clf.rgb):
              ex.color_picker(f'{c}', value=rgb2hex(rgb))
          v['color_clf'] = clf
        case 'Count':
          v['names'] = names
        case 'LineAndZone':
          v['draw']['lines'] = draw.lines
          v['draw']['zones'] = draw.zones
          v['reso'] = reso

    export = {
      'reso': reso,
      'preprocessors': preprocessors,
      'config': to_plain(config_plain),
      'model': asdict(model.info),
    }

    cmd = f'{Path(__file__).parent}/native.py --source {source}'
    c1, c2 = st.columns([1, 3])
    c2.subheader(f"Native run on {source if source != 0 else 'camera'}")

    match c2.radio(' ', ('Realtime inference', 'Save to video'), label_visibility='collapsed'):
      case 'Realtime inference':
        exe_button(c1, 'Show with OpenCV', cmd)
      case 'Save to video':
        saveto = c1.text_input(' ', 'result.mp4', label_visibility='collapsed')
        exe_button(c1, 'Save with OpenCV', f'{cmd} --saveto {saveto}')
    if c1.button('Save config to json'):
      with open('config.json', 'w') as f:
        json.dump(export, f, indent=2)

    config = from_plain(config_plain)
    anns = {i: all_class[i](**config[i]) for i in config}

    return cls(model=model, anns=anns)
