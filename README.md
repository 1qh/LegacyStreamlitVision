# Computer Vision Web UI

## Setup

#### Local environment

<details><summary>Note</summary>

- For non-GPU users, please install CPU version of PyTorch first

```
pip install -i https://download.pytorch.org/whl/cpu torch torchvision
```

</details>

```
pip install -r requirements.txt
```

| Streamlit (default)    | LightningAI (require install) |
| ---------------------- | ----------------------------- |
| `streamlit run app.py` | `lightning run app app.py`    |

#### Run in docker

```sh
docker-compose up
```

## Features

- Run locally on [Streamlit](https://github.com/streamlit/streamlit) / [LightningAI](https://github.com/lightning-ai/lightning) web UI

  - Model

    - Object detection
    - Object segmentation
    - Pose estimation
    - Image classification

  - On

    - Image
    - Video
    - Webcam

  - With ability to

    - Turn tracking on/off
    - Adjust confidence threshold
    - Filter by class
    - Object motion path
    - Object color classification
    - Trim video

- Draw visual elements interactively

  - Line count (in/out)
  - Polygon zone count

- Customize visual elements

  - Toggle on/off

    - Box
    - Label
    - Mask
    - Area
    - Trail
    - Count
    - FPS

  - Adjust

    - Text size
    - Text color
    - Text padding
    - Text offset
    - Line thickness
    - Mask opacity
    - Trail length

- **PRODUCTION READY**

  - Save drawed visual elements & settings in JSON
  - Run inference with OpenCV standalone from saved JSON

<details><summary>Note</summary>

Camera (`/dev/video0`) & native run (`cv2.imshow()`) is not configured to run in docker (you can try to mount your own device)

### TODO

#### Supported models:

- [x] All YOLOv8 models (Detect, Segment, Pose, Classify)
  - [x] With BoT-SORT / ByteTrack object tracking

Object detection:

- [x] RT-DETR
- [x] YOLO-NAS
- [x] YOLOv5
  - [x] new v5u models
  - [x] legacy v5 models
- [x] YOLOv3

Instance Segmentation

- [x] SAM

</details>

<details><summary>Experimental Features</summary>
- Fisheye undistortion
</details>
