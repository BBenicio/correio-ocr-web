from image_prep import remove_noise
from utils import conditional_save, get_conditional_path
import numpy as np
import cv2
import os

def detect_main_body(image, output_path: str = None, temp_folder: str = None):
    '''Detect the main body of text, discarding margins and other elements.

    Args:
        image (cv2 image): black and white image to process
        output_path (str): path to write the output image to, does not save if equals None. default=None
        temp_folder (str): folder to write the intermediary files to, does not save if equals None. default=None

    Returns:
        image of the main body (cv2 image)
    '''
    blur = cv2.GaussianBlur(image, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 75))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    if temp_folder:
        save_to = os.path.join(temp_folder, 'main_body/dilate.png')
        cv2.imwrite(save_to, dilate)
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    cnt = cnts[-1] # select largest
    x, y, w, h = cv2.boundingRect(cnt)

    img_main = image[y:y+h, x:x+w]
    if output_path:
        cv2.imwrite(output_path, img_main)

    return img_main

def detect_columns(image, output_folder: str = None, temp_folder: str = None, verbose: bool = False):
    '''Detect columns of text in an image.

    Args:
        image (cv2 image): black and white image to process
        output_folder (str): path to folder in which to save the image of detected columns, does not save if equals None. default=None
        temp_folder (str): path to folder in which to intermediary images, does not save if equals None. default=None

    Returns:
        tuple[list, list]: a tuple containing two elements:
            1. A list of the detected column images
            2. A list of the rectangle coordinates for each column (x, y, w, h)
    '''
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    if temp_folder:
        save_to = os.path.join(temp_folder, 'columns/dilate.png')
        cv2.imwrite(save_to, dilate)
    
    def contains(columns: 'list[tuple[int, int, int, int]]', x: int, y: int, w: int, h: int):
        '''Checks if the rectangle is already fully contained in a column.
        
        Args:
            columns (list): coordinates of the columns already detected
            x (int): x coordinate in pixels to check
            y (int): y coordinate in pixels to check
            w (int): width in pixels of the rectangle
            h (int): height in pixels of the rectangle
        
        Returns:
            bool: True if the rectangle (x, y, w, h) is fully contained in a rectangle in the columns list
        '''
        for c in columns:
            x1, y1, w1, h1 = c
            if x >= x1 and x + w <= x1 + w1 and y >= y1 and y + h <= y1 + h1:
                return True
        
        return False

    column_images = []
    boxed_image = image.copy() if temp_folder else None
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    
    if verbose:
        print(f'found {len(cnts)} contours, filtering...')

    i = 0
    columns = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h > 200 and w > 150 and not contains(columns, x, y, w, h):
            columns.append((x, y, w, h))
            roi = image[y:y+h, x:x+w]
            
            if output_folder: 
                save_to = os.path.join(output_folder, f'roi_{i}.png')
                cv2.imwrite(save_to, roi)
            
            column_images.append(roi)
            
            if temp_folder:
                cv2.rectangle(boxed_image, (x, y), (x+w, y+h), (36, 255, 12), 2)

            i += 1
        
    if verbose:
        print(f'finished filtering, got {len(columns)} columns')
    
    if temp_folder:
        save_to = os.path.join(temp_folder, 'column_boxes.png')
        cv2.imwrite(save_to, boxed_image)
    
    return column_images, columns


def crop_margins(image, temp_folder: str = None, output_path: str = None) -> tuple:
    '''Crop image to margins using line detection.

    Args:
        image (cv2 image): black and white image to process
        output_path (str): path to write the output image to, does not save if equals None. default=None
        temp_folder (str): folder to write the intermediary files to, does not save if equals None. default=None

    Returns:
        tuple: image of the main body (cv2 image); crop coordinates
    '''
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    erode = cv2.erode(thresh, kernel, iterations=1)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 7  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 500  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(erode, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    grouped_lines = ([], [])
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2 - y1) > image.shape[0] * 0.5 and abs(y2-y1) < image.shape[0] * 0.9:
                if x1 < image.shape[1] * 0.5: grouped_lines[0].append((x1, y1, x2, y2))
                else: grouped_lines[1].append((x1, y1, x2, y2))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    save_to = os.path.join(temp_folder, 'margins_lines.png') if temp_folder else None
    conditional_save(line_image, save_to)
    
    lines = []
    line = None
    h = 1e10
    for group in grouped_lines:
        if len(group) > 0:
            x = np.mean([np.mean([line[0], line[2]]) for line in group], dtype=int)
            y1 = np.min([min(line[1], line[3]) for line in group],)
            y2 = np.max([max(line[1], line[3]) for line in group])
            lines.append((x, y1, x, y2))
            if y2 - y1 < h:
                line = (x, y1, x, y2)
                h = y2-y1
    if len(lines) == 0:
        x1, x2 = 0, image.shape[1]
        y1, y2 = 0, image.shape[0]
    elif len(lines) == 1:
        if lines[0][0] < 0.5 * image.shape[1]:
            x1, x2 = 0, lines[0][0]
        else:
            x1, x2 = lines[0][0], image.shape[1]
        y1 = min(line[1], line[3])
        y2 = max(line[1], line[3])
    else:
        x1 = min(lines[0][0], lines[1][2])
        x2 = max(lines[0][0], lines[1][2])
        y1 = min(line[1], line[3])
        y2 = max(line[1], line[3])
    
    content = image[y1:y2, x1:x2]
    
    conditional_save(content, output_path)
    
    return content, (x1, x2, y1, y2)

def crop_background(image, temp_folder: str = None, output_path: str = None) -> tuple:
    '''Crop image removing background filtering color.

    Args:
        image (cv2 image): black and white image to process
        output_path (str): path to write the output image to, does not save if equals None. default=None
        temp_folder (str): folder to write the intermediary files to, does not save if equals None. default=None

    Returns:
        tuple: image of the main body (cv2 image); crop coordinates
    
    Remarks:
        reference https://medium.com/featurepreneur/colour-filtering-and-colour-pop-effects-using-opencv-python-3ce7d4576140
    '''
    # convert the BGR image to HSV colour space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bg = (95, 40, 120)
    upper_bg = (115, 90, 240)

    # create a mask for green colour using inRange function
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    
    save_to = os.path.join(temp_folder, 'bg_mask.png') if temp_folder else None
    conditional_save(mask, save_to)

    no_noise = remove_noise(mask, kernel_size=(50, 50))
    save_to = os.path.join(temp_folder, 'bg_mask_no_noise.png') if temp_folder else None
    conditional_save(no_noise, save_to)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    no_noise = cv2.dilate(no_noise, kernel, iterations=1)
    save_to = os.path.join(temp_folder, 'bg_mask_dilate.png') if temp_folder else None
    conditional_save(no_noise, save_to)

    no_noise = remove_noise(mask, kernel_size=(200, 200))
    save_to = os.path.join(temp_folder, 'mask_no_noise.png') if temp_folder else None
    conditional_save(no_noise, save_to)

    _, no_noise = cv2.threshold(no_noise, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts = cv2.findContours(no_noise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    cnt = cnts[-1] # select largest
    x, y, w, h = cv2.boundingRect(cnt)

    x1 = x
    x2 = x1+w
    y1 = y
    y2 = y1+h

    image = image[y1:y2, x1:x2]
    conditional_save(image, output_path)

    return image, (x1, x2, y1, y2)


def crop_to_page(image, temp_folder: str = None, output_path: str = None) -> tuple:
    '''Crop image to focus only on target page.

    Args:
        image (cv2 image): black and white image to process
        output_path (str): path to write the output image to, does not save if equals None. default=None
        temp_folder (str): folder to write the intermediary files to, does not save if equals None. default=None

    Returns:
        tuple: image of the main body (cv2 image); crop coordinates
    '''
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    blur = cv2.medianBlur(thresh, 13)
    conditional_save(blur, get_conditional_path('median_blur.png', temp_folder))
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 7  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    line_image = blur.copy() * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(blur, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    xs, ys = [], [] # list of x and y for long lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(y2 - y1) > line_image.shape[0] * 0.5 or abs(x2-x1) > line_image.shape[1] * 0.5:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
    conditional_save(line_image, get_conditional_path('lines.png', temp_folder))
    
    left = int(np.median([x for x in xs if x < thresh.shape[1] * 0.2])) if len(xs) > 0 else 0
    right = int(np.median([x for x in xs if x > thresh.shape[1] * 0.8])) if len(xs) > 0 else thresh.shape[1]
    top = int(np.median([y for y in ys if y < thresh.shape[0] * 0.2])) if len(ys) > 0 else 0
    bottom = int(np.median([y for y in ys if y > thresh.shape[0] * 0.8])) if len(ys) > 0 else thresh.shape[0]
    
    cv2.rectangle(line_image, (left,top), (right,bottom), 128, 2)
    conditional_save(line_image, get_conditional_path('lines_rect.png', temp_folder))
    
    cropped = image[top:bottom, left:right]
    conditional_save(cropped, output_path)
    
    return cropped, (left, right, top, bottom)