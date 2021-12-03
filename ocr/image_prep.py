import cv2
import numpy as np
import os
from utils import conditional_save

def grayscale(image, save_to: str = None):
    '''Make the image grayscale.

    Args:
        image (cv2 image): base image to convert
        save_to (str): path to save the image, does not save if it equals None. default=None
    
    Returns:
        processed image in cv2 image format
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conditional_save(image, save_to)
    return image

def black_and_white(image, maxval: int = 255, block_size: int = 45, save_to: str = None):
    '''Make the image black and white using adaptive thresholding.

    Args:
        image (cv2 image): base image to convert
        maxval (int): maxval to pass to OpenCV, default=255
        block_size (int): size of the block to use when thresholding. Must be odd, default=45
        save_to (str): path to save the image, does not save if it equals None. default=None

    Returns:
        processed image in cv2 image format
    '''
    image = cv2.adaptiveThreshold(image, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 10)
    conditional_save(image, save_to)
    return image

def remove_noise(image, kernel_size: 'tuple[int, int]' = (1, 1), dilate_iterations: int = 1, erode_iterations: int = 1, median_blur_k: int = 3, save_to: str = None):
    '''Remove noisy pixels from an image.

    Args:
        image (cv2 image): base image to process
        kernel_size (int): kernel_size to pass to OpenCV, default=(1,1)
        dilate_iterations (int): iterations for the dilation process to pass to OpenCV, default=1
        erode_iterations (int): iterations for the erosion process to pass to OpenCV, default=1
        median_blur_k (int): k-size for the medianBlur to passo to OpenCV, default=3
        save_to (str): path to save the image, does not save if it equals None. default=None

    Returns:
        processed image in cv2 image format
    '''
    kernel = np.ones(kernel_size, np.uint8)
    image = cv2.dilate(image, kernel, iterations=dilate_iterations)
    image = cv2.erode(image, kernel, iterations=erode_iterations)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, median_blur_k)
    conditional_save(image, save_to)
    return image

def prepare_image(image, output_path: str = None, temp_folder: str = None, binarize: bool = True, rotate: bool = True, remove_noise: bool = False, verbose: bool = False):
    '''
    Apply selected preparations to an image.

    Args:
        image (cv2 image): base image to process
        output_path (str): path to save the final image, does not save if equals None default=None
        temp_folder (str): path to save intermediary images, does not save if equals None default=None
        binarize (bool): flag to convert the image to black and white, default=True
        remove_noise (bool): flag to remove noise from the image, default=False
        verbose (bool): print extra information to console?
    
    Returns:
        processed image in cv2 image format
    '''
    if binarize:
        save_to = os.path.join(temp_folder, 'grayscale.png') if temp_folder else None
        if verbose:
            print('converting to grayscale...', f'saving temp file to "{save_to}"' if save_to else '')
        image = grayscale(image, save_to)

        save_to = os.path.join(temp_folder, 'black_and_white.png') if temp_folder else None
        if verbose:
            print('converting to black and white...', f'saving temp file to "{save_to}"' if save_to else '')
        image = black_and_white(image, save_to=save_to)
    
    if rotate:
        save_to = os.path.join(temp_folder, 'rotate.png') if temp_folder else None
        if verbose:
            print('rotating...', f'saving temp file to "{save_to}"' if save_to else '')
        image = deskew(image)
        conditional_save(image, save_to)

    if remove_noise:
        save_to = os.path.join(temp_folder, 'remove_noise.png') if temp_folder else None
        if verbose:
            print('removing pixel noise...', f'saving temp file to "{save_to}"' if save_to else '')
        image = remove_noise(image, save_to)
    
    if output_path:
        if verbose:
            print(f'saving final image to "{output_path}"')
        cv2.imwrite(output_path, image)
    
    return image


def get_skew_angle(cvImage) -> float:
    '''Get the angle to which an image is skewed.

    Args:
        cvImage (cv2 image): image to find the skew angle
    
    Returns:
        float: skew angle in degrees
    
    Remarks:
        https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    '''
    # Prep image, copy, convert to gray scale, blur, and threshold
    blur = cv2.GaussianBlur(cvImage, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    # dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    # contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    # largestContour = contours[0]
    # minAreaRect = cv2.minAreaRect(largestContour)
    angles = np.zeros(len(contours))
    # org = np.zeros(len(contours))
    for i in range(len(contours)):
        minAreaRect = cv2.minAreaRect(contours[i])
        
        angles[i] = minAreaRect[-1]# % 90 if minAreaRect[-1] > 0 else minAreaRect[-1] % -90
        # org[i] = minAreaRect[-1]
        if angles[i] < 0:
            angles[i] = 360 + angles[i]
        angles[i] = angles[i] % 90
        if angles[i] > 45:
            angles[i] = -90 + angles[i]
        angles[i] = -angles[i]
        # if angles[i] < -45:
            # angles[i] = 90 + angles[i]
        # angles[i] = -1.0 * angles[i]
    # print('', np.mean(org), np.quantile(org, [0, 0.25, 0.5, 0.75, 1]))
    # print('', np.mean(angles), np.quantile(angles, [0, 0.25, 0.5, 0.75, 1]))
    return np.median(angles)
    
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    # angle = minAreaRect[-1]
    # if angle < -45:
    #     angle = 90 + angle
    # return -1.0 * angle


def rotate_image(cvImage, angle: float):
    '''Rotate the image around its center.
    
    Args:
        cvImage (cv2 image): image to rotate
        angle (float): angle to rotate image by

    Returns:
        cv2 image: rotated image
    
    Remarks:
        https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
    '''
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew(cvImage):
    '''Deskew image

    Args:
        cvImage (cv2 image): image to deskew
    
    Returns:
        cv2 image: image rotated to be upright
    '''
    angle = get_skew_angle(cvImage)
    return rotate_image(cvImage, -1.0 * angle) if angle > -35 and angle < 35 else cvImage
