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
from ultralytics.engine.results import Boxes, Results
from vidgear.gears import VideoGear

from utils import cvt, filter_by_vals

coconames = YOLO().names


class LegacyYoloV5:
  def __init__(
    self,
    source: str | int,
    classes: list[int],
    conf: float,
    iou: float,
  ):
    self.model = yolov5.load(source)
    self.model.classes = classes
    self.model.conf = conf
    self.model.iou = iou

  def __call__(self, f: ndarray) -> list[Results]:
    res = Results(orig_img=f, path=None, names=self.model.names)
    pred = self.model(f).pred[0]
    res.boxes = Boxes(pred, f.shape)
    return [res]


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
    'options',
    'preprocessors',
    'task',
    'tracker',
  )

  def __init__(
    self,
    info: ModelInfo = ModelInfo(),
  ):
    classes = info.classes
    task = info.task
    conf = info.conf
    iou = info.iou
    tracker = info.tracker
    path = info.path
    ver = info.ver

    options = {}
    legacy = ver == 'v5'
    options = dict(
      classes=classes,
      conf=conf,
      iou=iou,
      retina_masks=True,
    )
    if legacy:
      model = LegacyYoloV5(path, classes, conf, iou)
      names = model.model.names
      options = {}
    else:
      if tracker:
        options.update(tracker=f'{tracker}.yaml', persist=True)
      match ver:
        case 'sam':
          model = SAM(path)
          names = []
        case 'rtdetr':
          model = RTDETR(path)
          names = coconames
        case 'NAS':
          model = NAS(path)
          names = model.model.names
        case _:
          model = YOLO(path)
          names = model.names
      model = model.predict if tracker is None else model.track

    names = names or coconames
    self.names = names
    self.task = task
    self.tracker = tracker
    self.info = info
    self.options = options
    self.model = model
    self.legacy = legacy
    self.preprocessors: list[callable] = []

  def __call__(self, source: str | int) -> Generator:
    stream = VideoGear(source=source).start()
    return self.gen(stream)

  def gen(self, stream: VideoGear) -> Generator:
    while (f := stream.read()) is not None:
      if self.preprocessors:
        for p in self.preprocessors:
          f = p(f)
      yield f, self.from_frame(f)

  def from_frame(self, f: ndarray) -> tuple[Detections, ndarray]:
    res = self.model(f, **self.options)[0]
    det = Detections.from_ultralytics(res) if res.boxes is not None else Detections.empty()
    fallback = np.zeros((1, 1, 3)) if self.legacy else cvt(res.plot(line_width=1, kpt_radius=1))
    return det, fallback

  def predict_image(self, file: str | bytes | Path):
    f = np.array(Image.open(file))
    if self.legacy:
      det = Detections.from_ultralytics(self.model(f)[0])
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
        c1, c2, c3 = ex.columns([1, 2, 1] if custom else [2, 2, 3])

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
        size = ex.selectbox('Size', ('mobile_sam', 'sam_b', 'sam_l'))
        path = f'{size}.pt'
        model = SAM(path)

    if ver != 'sam':
      if track:
        tracker = (
          ex.selectbox('Tracker', ['bytetrack', 'botsort', 'No track'])
          if task != 'classify'
          else None
        )
        tracker = tracker if tracker != 'No track' else None
      classes = filter_by_vals(
        coconames if ver == 'rtdetr' else model.model.names,
        ex,
        'Custom Classes',
      )
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
