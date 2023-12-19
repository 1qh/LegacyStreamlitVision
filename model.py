from glob import glob
from pathlib import Path
from typing import Generator

import numpy as np
import streamlit as st
import yolov5
from attrs import define
from numpy import ndarray
from PIL import Image
from streamlit import sidebar as sb
from supervision import BoxAnnotator, Detections
from ultralytics import NAS, RTDETR, SAM, YOLO
from vidgear.gears import VideoGear

from utils import cvt, filter_by_vals, legacy_generator


@define
class ModelInfo:
  path: str = 'yolov8n.pt'
  classes: list[int] = []
  ver: str = 'v8'
  task: str = 'detect'
  conf: float = 0.25
  iou: float = 0.5
  tracker: str | None = None


class Model:
  __slots__ = (
    'classes',
    'conf',
    'info',
    'iou',
    'legacy',
    'model',
    'names',
    'task',
    'tracker',
  )

  def __init__(
    self,
    info: ModelInfo = ModelInfo(),
  ):
    self.classes = info.classes
    self.task = info.task
    self.conf = info.conf
    self.iou = info.iou
    self.tracker = info.tracker

    path = info.path
    ver = info.ver

    self.legacy = ver == 'v5'

    match ver:
      case 'sam':
        self.model = SAM(path)
        self.names = []
      case 'rtdetr':
        self.model = RTDETR(path)
        self.names = []  # not available
      case 'NAS':
        self.model = NAS(path)
        self.names = self.model.model.names
      case _:
        self.model = yolov5.load(path) if self.legacy else YOLO(path)
        self.names = self.model.names

    if self.legacy:
      self.model.classes = self.classes
      self.model.conf = self.conf
      self.model.iou = self.iou

    self.info = info

  def __call__(self, source: str | int) -> Generator:
    if self.legacy:
      stream = VideoGear(source=source).start()
      return legacy_generator(stream, self.model)
    return (
      self.model.predict(
        source,
        stream=True,
        classes=self.classes,
        conf=self.conf,
        iou=self.iou,
        retina_masks=True,
      )
      if self.tracker is None
      else self.model.track(
        source,
        stream=True,
        classes=self.classes,
        conf=self.conf,
        iou=self.iou,
        retina_masks=True,
        tracker=f'{self.tracker}.yaml',
        persist=True,
      )
    )

  def from_res(self, res) -> tuple[Detections, ndarray]:
    if self.legacy:
      return Detections.from_yolov5(res[1]), np.zeros((1, 1, 3))

    if res.boxes is not None:
      return Detections.from_ultralytics(res), cvt(res.plot())

    return Detections.empty(), cvt(res.plot())

  def gen(self, source: str | int) -> Generator:
    for res in self(source):
      f = res[0] if self.legacy else res.orig_img
      yield f, self.from_res(res)

  def from_frame(self, f: ndarray) -> tuple[Detections, ndarray]:
    return self.from_res(
      (None, self.model(f))
      if self.legacy
      else self.model.predict(
        f,
        classes=self.classes,
        conf=self.conf,
        iou=self.iou,
        retina_masks=True,
      )[0]
    )

  def predict_image(self, file: str | bytes | Path):
    f = np.array(Image.open(file))
    if self.legacy:
      det = Detections.from_yolov5(self.model(f))
      f = BoxAnnotator().annotate(
        scene=f,
        detections=det,
        labels=[f'{conf:0.2f} {self.names[cls]}' for _, _, conf, cls, _ in det],
      )
    else:
      f = cvt(self.from_frame(f)[1])
    st.image(f)

  @classmethod
  def ui(cls, track: bool = True):  # sourcery skip: low-code-quality
    ex = sb.expander('Model', expanded=True)
    tracker = None

    match ex.radio(
      ' ',
      ('YOLO', 'RT-DETR', 'SAM'),
      horizontal=True,
      label_visibility='collapsed',
    ):
      case 'YOLO':
        suffix = {
          'Detect': '',
          'Segment': '-seg',
          'Classify': '-cls',
          'Pose': '-pose',
        }
        custom = ex.toggle('Custom weight')
        c1, c2 = ex.columns(2)
        c3, c4 = ex.columns(2)

        ver = c1.selectbox(
          'Version',
          ('v8', 'NAS', 'v5u', 'v5', 'v3'),
          label_visibility='collapsed',
        )
        legacy = ver == 'v5'
        is_nas = ver == 'NAS'
        sizes = ('n', 's', 'm', 'l', 'x')
        has_sizes = ver != 'v3'
        has_tasks = ver == 'v8'

        size = (
          c2.selectbox(
            'Size',
            sizes[1:4] if is_nas else sizes,
            label_visibility='collapsed',
          )
          if has_sizes and not custom
          else ''
        )
        task = (
          c3.selectbox(
            'Task',
            list(suffix.keys()),
            label_visibility='collapsed',
          )
          if has_tasks and not custom
          else 'detect'
        )
        if custom:
          path = c2.selectbox(' ', glob('*.pt'), label_visibility='collapsed')
        else:
          v = '_nas_' if is_nas else ver[:2]
          s = size if has_sizes else ''
          t = suffix[task] if has_tasks else ''
          u = ver[2] if len(ver) > 2 and ver[2] == 'u' else ''
          path = f'yolo{v}{s}{t}{u}.pt'

        if legacy:
          model = yolov5.load(path)
        else:
          if is_nas:
            try:
              model = NAS(path)
            except FileNotFoundError:
              st.warning(
                'You might want to go to https://docs.ultralytics.com/models to download the weights first.'
              )
          else:
            model = YOLO(path)
            task = model.overrides['task']
            path = model.ckpt_path

            if track:
              tracker = (
                c4.selectbox(
                  'Tracker',
                  ['bytetrack', 'botsort', 'No track'],
                  label_visibility='collapsed',
                )
                if task != 'classify'
                else None
              )
              tracker = tracker if tracker != 'No track' else None

          if custom:
            c3.subheader(f'{task.capitalize()}')

      case 'RT-DETR':
        ver = 'rtdetr'
        task = 'detect'
        size = ex.selectbox('Size', ('l', 'x'))
        path = f'{ver}-{size}.pt'
        model = RTDETR(path)
      case 'SAM':
        ver = 'sam'
        task = 'segment'
        size = ex.selectbox('Size', ('b', 'l'))
        path = f'{ver}_{size}.pt'
        model = SAM(path)

    if ver != 'sam':
      classes = filter_by_vals(model.model.names, ex, 'Custom Classes')
      conf = ex.slider('Threshold', max_value=1.0, value=0.25)
      iou = ex.slider('IoU', max_value=1.0, value=0.5)
    else:
      classes = []
      conf = 0.25
      iou = 0.5

    return cls(
      ModelInfo(
        path=path,
        classes=classes,
        ver=ver,
        task=task,
        conf=conf,
        iou=iou,
        tracker=tracker,
      )
    )
