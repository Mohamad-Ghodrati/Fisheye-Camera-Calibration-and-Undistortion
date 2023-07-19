import cv2
import numpy as np
import os
import glob

class FisheyeCalibrate:
    """
    A class that uses cv2.fisheye to find the K and D parameters of fisheye cameras using chessboard images.

    Methods:
        calculate_parameters: Calculates the K and D parameters of the camera and returns them.
        undistort: Undistorts input images using the calculated K and D parameters.
                   Please refer to the docstring of the undistort method for further explanation.

    Example:
        calibrator = FisheyeCalibrate(checkerboard_size=(8, 6), images_dir='path/to/images', image_extension='jpg')

        # Calculate camera parameters
        K, D = calibrator.calculate_parameters()

        # Undistort an image
        undistorted_img = calibrator.undistort(image)
    """
    def __init__(self, checkerboard_size: tuple, images_dir: str, image_extension='jpg') -> None:
        """Initialize the FisheyeCalibrate object.

        Args:
            checkerboard_size (tuple, list): Size of the checkerboard used in the calibration images.
            images_dir (str): Directory path of the calibration images.
            image_extension (str): Extension of the calibration images. Defaults to 'jpg'.
        """
        self.checkerboard = checkerboard_size
        self.images_path = images_dir + os.path.sep + f'*.{image_extension}'
        self.K = np.zeros((3, 3))
        self.D = np.zeros((4, 1))
        self.DIM = None
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

    def calculate_parameters(self):
        """
            Calculate the camera parameters K and D using chessboard images.

            This method performs camera calibration using a set of chessboard images. It detects the
            chessboard corners and uses them to calculate the camera parameters K and D.

            Returns:
                tuple: A tuple containing the camera intrinsic matrix K and distortion coefficients D.

            Raises:
                FileNotFoundError: If no images are found in the specified directory.
                ValueError: If the images do not have the same size or no usable images are found.

            Example:
                calibrator = FisheyeCalibrate(checkerboard=(7, 7), images_path='/path_to_images')
                K, D = calibrator.calculate_parameters()
            """
        objp = np.zeros((1, self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = []
        imgpoints = []
        images = glob.glob(self.images_path)

        if len(images) == 0:
            raise FileNotFoundError("No images found in the specified directory.")

        for fname in images:
            img = cv2.imread(fname)
            if _img_shape is None:
                _img_shape = img.shape[:2]
                self.DIM = _img_shape[::-1]
            else:
                if _img_shape != img.shape[:2]:
                    raise ValueError("All images must share the same size.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray,
                                                     self.checkerboard,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.subpix_criteria)
                imgpoints.append(corners)

        n_ok = len(objpoints)
        print(f'Found {n_ok} usable images.')

        if n_ok == 0:
            raise ValueError("Couldn't find any usable images in the directory.")

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(n_ok)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                self.K,
                self.D,
                rvecs,
                tvecs,
                self.calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        return self.K, self.D

    def undistort(self, img, balance=0.0, dim2=None, dim3=None):
        """
        Undistort input images using the calculated K and D parameters.

        Note:
            This method is not a recommended way to undistort your images,
            as it calculates `map1` and `map2` for `cv2.remap` each time it is called.
            It is intended for testing your parameters.

        Args:
            img (np.ndarray): Distorted image.
            balance (float, optional): Balance parameter. Default is 0.0.
            dim2 (tuple, optional): Dimension of the box you want to keep after undistorting the image.
                                    Default is None, which uses the original image dimensions.
            dim3 (tuple, optional): Dimension of the final box where OpenCV will put the undistorted image.
                                    Default is None, which uses the original image dimensions.

        Raises:
            ValueError: If K and D parameters have not been calculated.
                        Call the 'calculate_parameters' method before using this method.
            ValueError: If the aspect ratio of the input image is not equal to the aspect ratio of the images
                        used in calibration.

        Returns:
            np.ndarray: Undistorted image.
        """

        if np.sum(self.K) == 0:
            raise ValueError("K and D parameters must be calculated first. Call the 'calculate_parameters' method.")

        dim1 = img.shape[:2][::-1]
        if dim1[0] / dim1[1] != self.DIM[0] / self.DIM[1]:
            raise ValueError("Image to undistort must have the same aspect ratio as the ones used in calibration.")

        if not dim2:
            dim2 = dim1
        if not dim3:
            dim3 = dim1
        
        scaled_K = self.K * dim1[0] / self.DIM[0]
        scaled_K[2][2] = 1.0

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K,
                                                                       self.D,
                                                                       dim2,
                                                                       np.eye(3),
                                                                       balance=balance)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img


if __name__ == '__main__':
    images_directory = r''  # NOTE use your own directory.
    calibrator = FisheyeCalibrate((12, 8), images_directory, 'jpg')
    calibrator.calculate_parameters()
    print(f'K: \n{calibrator.K} \nD:\n{calibrator.D}')
    image_path = r''  # NOTE use your own image path to test the resualt.
    image = cv2.imread(image_path)
    image_undistorted = calibrator.undistort(image, balance=1)
    cv2.imshow('undistorted image', image_undistorted)
    cv2.waitKey()
