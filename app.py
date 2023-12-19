from pathlib import Path
from shutil import which

import cv2
import streamlit as st
from av import VideoFrame
from lightning.app import LightningApp, LightningFlow
from lightning.app.frontend import StreamlitFrontend
from psutil import process_iter
from streamlit import session_state
from streamlit import sidebar as sb
from streamlit_webrtc import webrtc_streamer
from supervision import VideoInfo

from core import Annotator
from model import Model
from utils import cvt, hms, maxcam, st_config, trim_vid

_shape = None

if 'path' not in session_state:
  session_state['path'] = ''


def prepare(path: str, place):
  vid = VideoInfo.from_video_path(path)

  if which('ffmpeg'):
    if place.toggle('Trim'):
      length = int(vid.total_frames / vid.fps)
      begin, end = place.slider(
        ' ',
        value=(0, length),
        max_value=length,
        label_visibility='collapsed',
      )
      begin, end = hms(begin), hms(end)
      if place.button(f'Trim from {begin[3:]} to {end[3:]}'):
        path = trim_vid(path, begin, end)
        session_state['path'] = path
    else:
      session_state['path'] = path
  else:
    session_state['path'] = path


def main(state):
  st_config()
  file = sb.file_uploader(' ', label_visibility='collapsed')
  running = sb.toggle('Realtime inference', help='Slower than native')
  usecam = sb.toggle('Use camera')
  mt = st.empty()

  if usecam:
    file = None

    an = Annotator.ui(0)
    width, height = maxcam()

    cap = cv2.VideoCapture(0)
    codec = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(6, codec)
    cap.set(5, 30)
    cap.set(3, width)
    cap.set(4, height)

    while running:
      t1, t2 = mt.tabs(['Main', 'Fallback'])
      if an.unneeded:
        t1, t2 = t2, t1
      success, f = cap.read()
      if success:
        f, fallback = an.from_frame(f)
        t1.image(cvt(f))
        t2.image(fallback)
      else:
        break
    cap.release()

    def cam_stream(key, callback):
      webrtc_streamer(
        key=key,
        video_frame_callback=callback,
        media_stream_constraints={
          'video': {
            'width': {'min': width},
            'height': {'min': height},
          }
        },
      )

    def simplecam(frame):
      f = an.from_frame(frame.to_ndarray(format='bgr24'))[1]
      return VideoFrame.from_ndarray(f)

    # oh my god, it took me so long to realize the frame bigger through time
    def cam(frame):
      f = frame.to_ndarray(format='bgr24')
      global _shape
      if f.shape != _shape:
        _shape = f.shape
        an.linezone.update(f)
      f = cvt(an.from_frame(f)[0])
      return VideoFrame.from_ndarray(f)

    if an.unneeded:
      cam_stream('a', simplecam)
    else:
      cam_stream('b', cam)

  if file:
    ex = sb.expander('Uploaded file')
    if 'image' in file.type:
      ex.image(file)
      Model.ui(track=False).predict_image(file)

    elif 'video' in file.type:
      ex.video(file)
      path = f'up_{file.name}'
      with open(path, 'wb') as up:
        up.write(file.read())

      prepare(path, ex)
      path = session_state['path']
      vid = VideoInfo.from_video_path(path)
      reso = vid.resolution_wh
      total_frames = vid.total_frames

      ex.markdown(
        f"""
            - Video resolution: {'x'.join([str(i) for i in reso])}
            - Total frames: {total_frames}
            - FPS: {vid.fps}
            - Path: {path}
                """
      )
      an = Annotator.ui(path)

      count = 0

      while running:
        for f, fallback in an.gen(path):
          t1, t2 = mt.tabs(['Main', 'Fallback'])
          if an.unneeded:
            t1, t2 = t2, t1
          t1.image(cvt(f))
          t2.image(fallback)
          count += 1
          t1.progress(count / total_frames)

    else:
      sb.warning('Please upload image/video')


class App(LightningFlow):
  def configure_layout(self):
    return StreamlitFrontend(render_fn=main)

  def run(self):
    pass


lit = LightningApp(App())

running_apps = [i for i in [p.cmdline() for p in process_iter()] if 'run' in i]
this_process = next(p for p in running_apps if any(Path(__file__).stem in a for a in p))

if 'app' not in this_process:
  main('')
