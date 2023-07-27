# Computer Vision Web UI

## Setup

#### Local environment

<details><summary>(For non-GPU users)</summary>

-   Install CPU version of PyTorch first

```
pip install -i https://download.pytorch.org/whl/cpu torch torchvision
```

</details>

```
pip install -r requirements.txt
```

| LightningAI                | Streamlit              |
| -------------------------- | ---------------------- |
| `lightning run app app.py` | `streamlit run app.py` |

#### Run in docker

```sh
docker-compose up
```

## Features

-   Run locally on [LightningAI](https://github.com/lightning-ai/lightning) / [Streamlit](https://github.com/streamlit/streamlit)

    -   Model

        -   Object detection
        -   Object segmentation
        -   Pose estimation
        -   Image classification

    -   On

        -   Image
        -   Video
        -   Webcam

    -   With ability to

        -   Turn tracking on/off
        -   Adjust confidence threshold
        -   Filter by class
        -   Object motion path
        -   Object color classification
        -   Trim video

-   Draw visual elements interactively

    -   Line count (in/out)
    -   Polygon zone count

-   Customize visual elements

    -   Toggle on/off

        -   Box
        -   Label
        -   Mask
        -   Area
        -   Trail
        -   Count
        -   FPS

    -   Adjust

        -   Text size
        -   Text color
        -   Text padding
        -   Text offset
        -   Line thickness
        -   Mask opacity
        -   Trail length

-   **PRODUCTION READY**

    -   Save drawed visual elements & settings in JSON
    -   Run inference with OpenCV standalone from saved JSON

<details><summary>Note</summary>

Camera (`/dev/video0`) & native run (`cv2.imshow()`) is not configured to run in docker (you can try to mount your own device)

### TODO

#### Supported models:

-   [x] All YOLOv8 models (Detect, Segment, Pose, Classify)
    -   [x] With tracking

Object detection:

-   [x] RT-DETR
-   [x] YOLO-NAS
-   [x] YOLOv5
    -   [x] new v5u models
    -   [x] original v5 models
-   [x] YOLOv3

Instance Segmentation

-   [x] SAM

</details>
