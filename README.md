# Fisheye Camera Calibration and Undistortion
This repository is based on a medium article named [Calibrate fisheye lens using OpenCV — part 1 & 2](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0) by [Kenneth Jiang](https://www.linkedin.com/in/kennethjiang/).

This repository provides a Python implementation of fisheye camera calibration and image undistortion using the OpenCV library. 
The FisheyeUndistort class in this repository utilizes OpenCV's **`cv2.fisheye`** module to find the camera parameters (**`K`** and **`D`**) for fisheye cameras using chessboard calibration images. It also provides a method to undistort input images using the calculated camera parameters.
## Features
* Calculation of camera parameters (**`K`** and **`D`**) using chessboard calibration images.
* Handling of images with different aspect ratios compared to the calibration images.
* Easy-to-use interface with example usage.
## Requirements
* Python 3.x
* OpenCV (cv2) (**version 3.0 or higher**)
* Numpy
## Usage
Here is an example of how to use the **`FisheyeUndistort`** class:

```python
import cv2
from fisheye_undistort import FisheyeUndistort

calibrator = FisheyeUndistort(checkerboard_size=(8, 6), images_dir='path/to/images', image_extension='jpg')

# Calculate camera parameters
K, D = calibrator.calculate_parameters()

# Undistort an image
image = cv2.imread('path/to/image.jpg')
undistorted_img = calibrator.undistort(image)

cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## References
* [Calibrate fisheye lens using OpenCV — part 1](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)
* [Calibrate fisheye lens using OpenCV — part 2](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f)
