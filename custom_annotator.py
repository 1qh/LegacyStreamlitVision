import numpy as np
from numpy import ndarray
from supervision import (
  Color,
  ColorPalette,
  Detections,
  FPSMonitor,
  LineZone,
  LineZoneAnnotator,
  Point,
  PolygonZone,
  PolygonZoneAnnotator,
  crop_image,
  draw_text,
  get_polygon_center,
)
from supervision.annotators.base import BaseAnnotator

from utils import ColorClassifier, Draw, avg_rgb, plur


class ColorClassifierAnnotator(BaseAnnotator):
  def __init__(
    self,
    color_clf: ColorClassifier = ColorClassifier(),
    naive: bool = False,
    text_color: Color = Color.black(),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 10,
  ):
    self.naive: bool = naive
    self.text_color: Color = text_color
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding
    if color_clf.names:
      self.rgb_colors = color_clf.rgb
      self.color_names = color_clf.names or []
      self.color_clf = color_clf

  def annotate(
    self,
    scene: ndarray,
    detections: Detections,
  ) -> ndarray:
    xyxy = detections.xyxy.astype(int)
    centers = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) // 2

    for center, bb in zip(centers, xyxy):
      x = center[0]
      y = center[1]

      # for shirt color of person
      # w = bb[2] - bb[0]
      # h = bb[3] - bb[1]
      # cropped = f[
      #     bb[1] : bb[3] - int(h * 0.4),
      #     bb[0] + int(w * 0.2) : bb[2] - int(w * 0.2),
      # ]

      cropped = crop_image(scene, bb)
      if 0 in cropped.shape:
        continue

      rgb = scene[y, x] if self.naive else avg_rgb(cropped)
      predict = self.color_clf.closest(rgb)
      r, g, b = self.rgb_colors[predict]
      draw_text(
        scene=scene,
        text=self.color_names[predict],
        text_anchor=Point(x=x, y=y + 20),
        text_color=Color(255 - r, 255 - g, 255 - b),
        text_scale=self.text_scale,
        text_padding=self.text_padding,
        background_color=Color(r, g, b),
      )
    return scene


class FpsAnnotator(BaseAnnotator):
  def __init__(
    self,
    text_anchor: Point = Point(x=50, y=20),
    text_color: Color = Color.black(),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 1,
    sample_size: int = 10,
  ):
    self.text_anchor: Point = text_anchor
    self.text_color: Color = text_color
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding
    self.fps_monitor = FPSMonitor(sample_size)

  def annotate(
    self,
    scene: ndarray,
    detections: Detections,
  ) -> ndarray:
    self.fps_monitor.tick()
    fps = self.fps_monitor()
    draw_text(
      scene=scene,
      text=f'{fps:.2f}',
      text_anchor=self.text_anchor,
      text_color=self.text_color,
      text_scale=self.text_scale * 2,
      text_thickness=self.text_thickness,
      text_padding=self.text_padding,
    )
    return scene


class CountAnnotator(BaseAnnotator):
  def __init__(
    self,
    names: list[str] = None,
    text_anchor: Point = Point(x=50, y=12),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 1,
  ):
    if names is None:
      names = []
    self.names = names
    self.text_anchor: Point = text_anchor
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding
    self.pallet: ColorPalette = ColorPalette.default()

  def annotate(
    self,
    scene: ndarray,
    detections: Detections,
  ) -> ndarray:
    names = self.names
    if len(names):
      class_ids = detections.class_id

      for i, c in enumerate(np.bincount(class_ids)):
        if c:
          bg = self.pallet.by_idx(i)
          draw_text(
            scene=scene,
            text=plur(c, names[i]),
            text_anchor=Point(
              x=scene.shape[1] - self.text_anchor.x,
              y=self.text_anchor.y + int(i * self.text_scale * 18),
            ),
            text_color=Color.white() if np.sum(bg.as_bgr()) < 384 else Color.black(),
            text_scale=self.text_scale,
            text_thickness=self.text_thickness,
            text_padding=self.text_padding,
            background_color=bg,
          )
    return scene


class AreaAnnotator(BaseAnnotator):
  def __init__(
    self,
    text_color: Color = Color.black(),
    text_scale: float = 0.5,
    text_thickness: int = 1,
    text_padding: int = 1,
  ):
    self.text_color: Color = text_color
    self.text_scale: float = text_scale
    self.text_thickness: int = text_thickness
    self.text_padding: int = text_padding

  def annotate(
    self,
    scene: ndarray,
    detections: Detections,
  ) -> ndarray:
    xyxy = detections.xyxy.astype(int)
    centers = (xyxy[:, [0, 1]] + xyxy[:, [2, 3]]) // 2
    for a, c in zip(detections.area.astype(int), centers):
      draw_text(
        scene=scene,
        text=str(a)[:-1],
        text_anchor=Point(x=c[0], y=c[1]),
        text_color=self.text_color,
        text_scale=self.text_scale,
        text_thickness=self.text_thickness,
        text_padding=self.text_padding,
      )
    return scene


class LineAndZoneAnnotator(BaseAnnotator):
  def __init__(
    self,
    draw: Draw = Draw(),
    reso: tuple[int, int] = (640, 480),
    thickness: int = 2,
    text_thickness: int = 2,
    text_color: Color = Color.black(),
    text_scale: float = 0.5,
    text_offset: float = 1.5,
    text_padding: int = 1,
  ):
    self.draw: Draw = draw
    self.reso: tuple[int, int] = reso
    self.ls: list[LineZone] = [
      LineZone(
        start=Point(i[0][0], i[0][1]),
        end=Point(i[1][0], i[1][1]),
      )
      for i in self.draw.lines
    ]
    self.line = LineZoneAnnotator(
      thickness=thickness,
      text_thickness=text_thickness,
      text_color=text_color,
      text_scale=text_scale,
      text_offset=text_offset,
      text_padding=text_padding,
    )
    self.zs: list[PolygonZone] = [
      PolygonZone(
        polygon=np.array(p, dtype=np.int32),
        frame_resolution_wh=self.reso,
      )
      for p in self.draw.zones
    ]
    self.origin_zs = self.zs
    self.zones: list[PolygonZoneAnnotator] = [
      PolygonZoneAnnotator(
        zone=z,
        color=ColorPalette.default().by_idx(i),
        thickness=thickness,
        text_color=text_color,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
      )
      for i, z in enumerate(self.zs)
    ]

  def annotate(
    self,
    scene: ndarray,
    detections: Detections,
  ) -> ndarray:
    for l in self.ls:  # noqa: E741
      l.trigger(detections)
      self.line.annotate(frame=scene, line_counter=l)

    for z, zone in zip(self.zs, self.zones):
      z.trigger(detections)
      zone.annotate(scene)

    return scene

  def update(self, f: ndarray):
    scale = f.shape[0] / self.reso[1]
    self.ls = [
      LineZone(
        start=Point(i[0][0] * scale, i[0][1] * scale),
        end=Point(i[1][0] * scale, i[1][1] * scale),
      )
      for i in self.draw.lines
    ]
    self.zs = [
      PolygonZone(
        polygon=(z.polygon * scale).astype(int),
        frame_resolution_wh=(f.shape[1], f.shape[0]),
      )
      for z in self.origin_zs
    ]
    for i, z in enumerate(self.zs):
      self.zones[i].zone = z
      self.zones[i].center = get_polygon_center(polygon=z.polygon)
