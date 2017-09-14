# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline.

**I/O**

Aside from the processing part other the pipeline needs to solve other problems
  - Load a video
  - Extract single frames
  - Combine processed frames to new video file

Load video:
The video file is defined and opened using cv2.VideoCapture.
```python
#video = "challenge.mp4"
video="solidYellowLeft.mp4"
#video="solidWhiteRight.mp4"
vidcap = cv2.VideoCapture(video)
success,image = vidcap.read()
```
While the reader is successfully reading more images from the video stream, each extracted frame is processed and added to the output stream. 

```python
success = True
while success:
    success,image = vidcap.read()
    print("processing Frame "+count)
    count=count+1
    if (init):
        writer = cv2.VideoWriter(video.replace(".mp4","_processed.mp4"),fourcc, fps, (image.shape[1],image.shape[0]))
        init=bool(0)
    if (success):
        writer.write(findLaneLines(image))
    else:
        writer.release()
```



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
