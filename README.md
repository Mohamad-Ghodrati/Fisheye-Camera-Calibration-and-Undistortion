# Fisheye Camera Calibration and Undistortion
This repository is based on a medium article named [Calibrate fisheye lens using OpenCV — part 1 & 2](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0) by [Kenneth Jiang](https://www.linkedin.com/in/kennethjiang/).

This repository provides a Python implementation of fisheye camera calibration and image undistortion using the OpenCV library. 

The **`FisheyeCalibrate`** class in this repository utilizes OpenCV's **`cv2.fisheye`** module to find the camera parameters (**`K`** and **`D`**) for fisheye cameras using chessboard calibration images. It also provides a method to undistort input images using the calculated camera parameters.

The **`FisheyeUndistort`** class in this repository is designed to undistort images captured by fisheye cameras using the camera parameters obtained from the **`FisheyeCalibrate`** class.



## Features
* Calculation of camera parameters (**`K`** and **`D`**) using chessboard calibration images.
* Handling of images with different aspect ratios compared to the calibration images.
* Easy-to-use interface with example usage.
## Requirements
* Python 3.x
* OpenCV (cv2) (**version 3.0 or higher**)
* Numpy
## Usage
Here is an example of how to use the **`FisheyeCalibrate`** and **`FisheyeUndistort`** class:

```python
import cv2
from fisheye_calibrate import FisheyeCalibrate
from fisheye_undistort import FisheyeUndistort

calibrator = FisheyeCalibrate(checkerboard_size=(12, 8), images_dir=r'', image_extension='jpg')

# Calculate camera parameters
K, D = calibrator.calculate_parameters()

 
calibration_image_size = calibrator.DIM

image = cv2.imread(r'')
input_size = image.shape[:2][::-1]

balance = 0.5
device = 'cpu'
fisheye_undistorter = FisheyeUndistort(K, D, calibration_image_size, input_size, balance, device)

# Undistort an image
undistorted_image = fisheye_undistorter.undistort(image)



cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


## References
* [Calibrate fisheye lens using OpenCV — part 1](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)
* [Calibrate fisheye lens using OpenCV — part 2](https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f)
