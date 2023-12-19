#!/usr/bin/env python3
import cv2
from typer import run
from vidgear.gears import WriteGear

from core import Annotator


def app(source=0, config='config.json', saveto=None):
  if '.' not in source and int(source) in range(-1, 2):
    source = int(source)

  gen = Annotator.load(config).gen(source)

  if saveto is None:
    for f, _ in gen:
      cv2.imshow('', f)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  else:
    writer = WriteGear(output=saveto)
    for f, _ in gen:
      writer.write(f)
    writer.close()

  cv2.destroyAllWindows()


if __name__ == '__main__':
  run(app)
