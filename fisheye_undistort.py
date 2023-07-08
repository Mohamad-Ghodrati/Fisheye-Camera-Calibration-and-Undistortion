import cv2
import numpy as np


class FisheyeUndistort:
    """
    A class for performing fisheye image undistortion.

    Attributes:
        K (np.ndarray): The intrinsic matrix (K) of the camera.
        D (np.ndarray): The distortion coefficients (D) of the camera.
        device (str): The device used for computation.
        map1 (np.ndarray or cv2.cuda_GpuMat): The first map for undistortion.
        map2 (np.ndarray or cv2.cuda_GpuMat): The second map for undistortion.
        __gpu_mat (cv2.cuda_GpuMat): GpuMat object for CUDA-based computation.

    Methods:
        calculate_map: Calculates the undistortion maps based on the provided parameters.
        undistort: Undistorts an input image using the precomputed maps.

    Example:
        intrinsic_matrix = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
        distortion_coeffs = np.array([0.1, -0.2, 0.3, -0.4])
        calibration_image_size = (640, 480)
        input_size = (1280, 960)
        balance = 0.5
        device = 'cuda'
        fisheye_undistorter = FisheyeCalibrate(intrinsic_matrix, distortion_coeffs, calibration_image_size, input_size, balance, device)

        # Undistort an image
        image = cv2.imread('image.jpg')
        undistorted_image = fisheye_undistorter.undistort(image)
    """
    def __init__(self, intrinsic_matrix, distortion_coeffs, calibration_image_size, input_size, balance, device, dim2=None, dim3=None):
        """
        Initializes the FisheyeCalibrate object with the specified parameters.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix (K) of the camera.
            distortion_coeffs (np.ndarray): The distortion coefficients (D) of the camera.
            calibration_image_size (tuple): Size of the calibration images used for camera calibration (w,h).
            input_size (tuple): Size of the input images to be undistorted (w,h).
            balance (float): Balance parameter for undistortion.
            device (str): The device to be used for computation. Supported options: 'cuda', 'cpu'.
            dim2 (tuple, optional): Dimension of the box you want to keep after undistorting the image.
                                    Default is None, which uses the input image dimensions (w,h).
            dim3 (tuple, optional): Dimension of the final box where OpenCV will put the undistorted image.
                                    Default is None, which uses the input image dimensions (w,h).
        """
        self.K = intrinsic_matrix
        self.D = distortion_coeffs
        self.device = device
        self.map1, self.map2 = self.calculate_map(calibration_image_size, input_size, balance, dim2, dim3)
        self.__gpu_mat = cv2.cuda_GpuMat()

    def calculate_map(self, DIM, dim1, balance, dim2, dim3):
        """
        Calculates the undistortion maps based on the provided parameters.

        Args:
            DIM (tuple): Size of the calibration images used for camera calibration.
            dim1 (tuple): Size of the input images to be undistorted.
            balance (float): Balance parameter for undistortion.
            dim2 (tuple, optional): Dimension of the box to keep after undistortion. Defaults to None (uses dim1).
            dim3 (tuple, optional): Dimension of the final box for the undistorted image. Defaults to None (uses dim1).

        Returns:
            tuple: The first and second undistortion maps.

        Raises:
            ValueError: If the aspect ratio of the input image is not equal to the aspect ratio of the calibration images.
        """
        if dim1[0] / dim1[1] != DIM[0] / DIM[1]:
            raise ValueError("Image to undistort must have the same aspect ratio as the ones used in calibration.")
        dim2 = dim2 if dim2 is not None else dim1
        dim3 = dim3 if dim3 is not None else dim1

        scaled_K = self.K * dim1[0] / DIM[0]
        scaled_K[2][2] = 1.0

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K,
                                                                       self.D,
                                                                       dim2,
                                                                       np.eye(3),
                                                                       balance=balance)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim3, cv2.CV_32F)
        if self.device == 'cuda':
            map1, map2 = list(map(cv2.cuda_GpuMat, (map1, map2)))

        return map1, map2

    def undistort(self, image):
        """
        Undistorts an input image using the precomputed undistortion maps.

        Args:
            image (np.ndarray): The input image to be undistorted.

        Returns:
            np.ndarray: The undistorted image.
        """
        if self.device == 'cuda':
            self.__gpu_mat.upload(image)
            undistorted = cv2.cuda.remap(self.__gpu_mat, self.map1, self.map2, cv2.INTER_LINEAR)
            undistorted = undistorted.download()
            return undistorted
        else:
            undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
            return undistorted


if __name__ == '__main__':
    K = ...
    D = ...
    fisheye_undistorter = FisheyeUndistort(K, D, (820, 616), (820, 616), 1.0, 'cpu')
    image = cv2.imread(r'')  
    cv2.imshow('image', image)
    undistorted_image = fisheye_undistorter.undistort(image)
    cv2.imshow('undistorted image', undistorted_image)

    cv2.waitKey()


