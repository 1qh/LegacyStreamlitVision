import pickle
from multiprocessing import Pool

import cv2
import numpy as np


def unit_vector(vector):
  np.seterr(invalid='ignore')
  norm = np.linalg.norm(vector)
  return vector / norm


def angle_between(v1, v2):
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_map(points):
  point, center, top_point, radius = points
  point_np = np.asarray(point)
  distance = np.linalg.norm(point_np - center)
  length_percent = None
  if distance <= radius:
    angle = angle_between(top_point - center, point_np - center)
    length_percent = angle / (2 * np.pi)
  return point, length_percent, distance


class FisheyeWarping:
  def __init__(self, img):
    self.img = img

    self.dewarp_map_x, self.dewarp_map_y = None, None
    self.rewarp_map_x, self.rewarp_map_y, self.rewarp_mask = None, None, None
    self.pano_shape = None

  def build_dewarp_mesh(self):
    self.dewarp_map_x, self.dewarp_map_y = self.build_dewarp_map(self.img)
    print(f'Dewarp Map X shape -> {self.dewarp_map_x.shape}')
    print(f'Dewarp Map Y shape -> {self.dewarp_map_y.shape}')
    h, w = self.dewarp_map_x.shape
    self.pano_shape = (w, h)
    return self.pano_shape, self.dewarp_map_x, self.dewarp_map_y

  def load_dewarp_mesh(self, mesh_path: str):
    with open(mesh_path, 'rb') as f:
      self.pano_shape, self.dewarp_map_x, self.dewarp_map_y = pickle.load(f)
    return self.pano_shape, self.dewarp_map_x, self.dewarp_map_y

  def run_dewarp(self):
    return self.dewarp(self.img, flip=True)

  def build_rewarp_mesh(self):
    self.rewarp_map_x, self.rewarp_map_y, self.rewarp_mask = self.build_rewarp_map()
    print(f'Rewarp Map X shape -> {self.rewarp_map_x.shape}')
    print(f'Rewarp Map Y shape -> {self.rewarp_map_y.shape}')
    print(f'Rewarp Map MASK shape -> {self.rewarp_mask.shape}')
    return self.rewarp_map_x, self.rewarp_map_y, self.rewarp_mask

  def load_rewarp_mesh(self, mesh_path: str):
    with open(mesh_path, 'rb') as f:
      self.rewarp_map_x, self.rewarp_map_y, self.rewarp_mask = pickle.load(f)
    return self.rewarp_map_x, self.rewarp_map_y, self.rewarp_mask

  def run_rewarp(self):
    dewarp_result = self.dewarp(self.img, flip=False)
    return self.rewarp(dewarp_result, flip=True)

  def run_rewarp_with_mesh(self, pano_img):
    warning = 'Rewarp needs the shape of pano generated from `run_dewarp`. Please run it first.'
    assert self.pano_shape is not None, warning
    pano_img = cv2.resize(pano_img, self.pano_shape)
    pano_img = self.wrap(pano_img, rotate_angle=180, scale=1)
    return self.rewarp(pano_img, flip=True)

  def get_fisheye_img_data(self, img):
    # Center
    c_x = int(img.shape[1] / 2)
    c_y = int(img.shape[0] / 2)
    # Inner circle, now is zero
    r1_x = int(img.shape[1] / 2)
    # r1_y = int(img.shape[0]/2)
    r1 = r1_x - c_x
    # Outter circle
    r2_x = int(img.shape[1])
    # r2_y = 0
    r2 = r2_x - c_x
    # Rectangle width and height
    w_d = int(2.0 * ((r2 + r1) / 2) * np.pi)
    h_d = r2 - r1
    return w_d, h_d, r1, r2, c_x, c_y

  def dewarp_map_job(self, point):
    y, x, img_details = point
    w_d, h_d, r1, r2, c_x, c_y = img_details
    r = (float(y) / float(h_d)) * (r2 - r1) + r1
    theta = (float(x) / float(w_d)) * 2.0 * np.pi
    x_s = int(c_x + r * np.sin(theta))
    y_s = int(c_y + r * np.cos(theta))
    return (y, x), x_s, y_s

  def build_dewarp_map(self, img):
    img_details = self.get_fisheye_img_data(img)
    w_d, h_d, _, _, _, _ = img_details
    mapx = np.zeros((h_d, w_d), np.float32)
    mapy = np.zeros((h_d, w_d), np.float32)

    jobList = []
    for y in range(int(h_d - 1)):
      jobList.extend((y, x, img_details) for x in range(int(w_d - 1)))
    with Pool() as p:
      results = p.map(self.dewarp_map_job, jobList)
    for point, x_s, y_s in results:
      mapx.itemset(point, x_s)
      mapy.itemset(point, y_s)

    return mapx, mapy

  def dewarp(self, img, flip=False):
    warning = 'Dewarp mesh have not been created! Please run `build_dewarp_mesh` first.'
    assert self.dewarp_map_x is not None, warning
    assert self.dewarp_map_y is not None, warning
    output = cv2.remap(img, self.dewarp_map_x, self.dewarp_map_y, cv2.INTER_LINEAR)
    if flip:
      output = self.wrap(output, 180, scale=1)
    return output

  def wrap(self, img, rotate_angle, scale=1 / 3):
    h, w = img.shape[:2]
    # get the center
    center = (w / 2, h / 2)
    r_matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1)
    img = cv2.warpAffine(img, r_matrix, (w, h))
    img = cv2.resize(img, dsize=(int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

  def build_rewarp_map(self):
    width, height, _ = self.img.shape
    xmap = np.zeros((width, height), dtype=np.float32)
    ymap = np.zeros((width, height), dtype=np.float32)
    mask = np.zeros((width, height), dtype=np.uint8)
    center = np.asarray([int(width / 2), int(height / 2)])
    top_point = np.asarray([int(width / 2), 0])
    radius = width / 2
    jobs = []
    for y, channel in enumerate(self.img):
      for x, _ in enumerate(channel):
        points = ((x, y), center, top_point, radius)
        jobs.append(points)
    print('Start multi pixel mapping for rewarpping')
    with Pool() as p:
      results = p.map(angle_map, jobs)
    dewarp_result = self.dewarp(self.img, flip=False)
    for result in results:
      (x, y), length_percentage, distance = result
      if length_percentage is not None:
        point = (y, x)
        length = length_percentage * dewarp_result.shape[1]
        xmap.itemset(point, length)
        ymap.itemset(point, distance)
        mask.itemset(point, 255)

    return xmap, ymap, mask

  def remap(self, img, x, y):
    return cv2.remap(img, x, y, cv2.INTER_LINEAR)

  def half_rewarp_map(self, pano_img, x, y):
    # get left part
    left_output = self.remap(pano_img, x, y)
    # get right part
    right_output = self.remap(cv2.flip(pano_img, 1), x, y)
    return left_output, right_output

  def rewarp(self, pano_img, flip=False):
    warning = 'Rewarp mesh have not been created! Please run `build_rewarp_mesh` first.'
    assert self.rewarp_map_x is not None, warning
    assert self.rewarp_map_y is not None, warning
    assert self.rewarp_mask is not None, warning
    left_output, right_output = self.half_rewarp_map(pano_img, self.rewarp_map_x, self.rewarp_map_y)

    re_render_canvas = left_output
    # find the center of the image
    vertical_center = int(re_render_canvas.shape[0] / 2) + 1
    # combine 2 parts
    re_render_canvas[:, vertical_center:, :] = right_output[:, vertical_center:, :]

    re_render_canvas = cv2.add(
      re_render_canvas,
      np.zeros(re_render_canvas.shape, dtype=np.uint8),
      mask=self.rewarp_mask,
    )

    if flip:
      re_render_canvas = self.wrap(re_render_canvas, 180, scale=1)

    return re_render_canvas
