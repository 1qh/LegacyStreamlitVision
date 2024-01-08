#!/usr/bin/env python3
from os import system
from subprocess import check_output

from typer import Argument, run
from typing_extensions import Annotated


def get_wh(path: str) -> tuple[int, int]:
  return list(
    map(
      int,
      check_output(
        f'ffprobe -v error -show_entries stream=width,height -of csv=p=0 {path}',
        shell=True,
      )
      .decode()
      .strip()
      .split(','),
    )
  )


def app(
  origin: Annotated[str, Argument(help='Path to original fisheye video')],
  result: Annotated[str, Argument(help='Path to result flatten video')],
):
  crop = min(get_wh(origin))
  h = min(get_wh(result))
  cmd = f"""mpv {origin} --external-file={result} --lavfi-complex='[vid1] crop={crop}:{crop} [cropped]; [cropped] scale={h}x{h} [scaled]; [scaled][vid2] hstack [vo]'"""
  print(cmd)
  system(cmd)


if __name__ == '__main__':
  run(app)
