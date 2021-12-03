import cv2
import numpy as np
from utils import conditional_save, get_conditional_path

def cc_analisys(img) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    '''Find connected components and extract features from them.

    Get the connected components and their: area, density, bounding box, inner
    CCs and height/width rate.

    Args:
        img (cv2 image): inverse binary image.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: area,
        density, bounding box, number of inner CCs and height/width rate for each
        connected component.
    '''
    n, _, cc, _ = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)
    ### Análise dos Componentes Conexos
    area = np.zeros(n, dtype=np.int)
    density = np.zeros(n, dtype=np.float)
    rect = np.zeros((n, 4), dtype=np.int)
    inc = np.zeros(n, dtype=np.int)
    hw_rate = np.zeros(n)
    for i in range(1, n):
        h = cc[i, cv2.CC_STAT_HEIGHT]
        w = cc[i, cv2.CC_STAT_WIDTH]
        area[i] = cc[i, cv2.CC_STAT_AREA]
        density[i] = area[i] / (w*h)
        hw_rate[i] = min(w, h) / max(w, h)
        rect[i, [0,1]] = cc[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP]]
        rect[i, [2,3]] = [w, h]
    for i in range(1, n):
        contained = (rect[:, 0] >= rect[i, 0]) & (rect[:, 0] + rect[:, 2] <= rect[i, 0] + rect[i, 2]) & (rect[:, 1] >= rect[i, 1]) & (rect[:, 1] + rect[:, 3] <= rect[i, 1] + rect[i, 3])
        contained[i] = False
        contained = contained & (area >= area[i] * 0.05)
        inc[i] = contained.sum()
    
    return area, density, rect, inc, hw_rate


def heuristic_filter(img, area: np.ndarray, density: np.ndarray, rect: np.ndarray, inc: np.ndarray, hw_rate: np.ndarray) -> 'tuple[np.ndarray, np.ndarray]':
    ''' Apply a heuristic filter to remove non-text elements from an image.

    Use the heuristic filter defined by (Tran et al. 2017) to identify and
    remove non-text elements from an image.

    Args:
        img (cv2 image): inverse binary image
        area (np.ndarray): areas (number of pixels) of the CCs
        density (np.ndarray): density of the CCs
        rect (np.ndarray): bounding boxes of the CCs
        inc (np.ndarray): number of contained CCs
        hw_rate (np.ndarray): height/width rate of the CCs

    Returns:
        tuple[np.ndarray, np.ndarray]: the image without the non-text elements,
        and a boolean mask for the text CCs.
    '''
    is_text = np.full(rect.shape[0], True, dtype=np.bool8)

    is_text = is_text & (rect[:, 0] > 0)
    is_text = is_text & (rect[:, 1] > 0)
    is_text = is_text & (rect[:, 0] + rect[:, 2] < img.shape[1])
    is_text = is_text & (rect[:, 0] + rect[:, 3] < img.shape[0])

    is_text = is_text & (inc <= 4)
    is_text = is_text & (area >= 20)
    is_text = is_text & ~((hw_rate < 0.1))# & (rect[:, 3] < rect[:, 2]))
    is_text = is_text & (density >= 0.06)
    # is_text = is_text & (density <= 0.9)

    out = img.copy() * 0
    for x,y,w,h in rect[is_text]:
        out[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    return out, is_text


def get_gradient(R, s: int, axis: int = 1, t: int = 0) -> np.ndarray:
    '''Calculate the gradient for the projection on the image

    Using the method outlined in (Tran et al. 2016), calculate the gradient of
    the horizontal (axis=0) or vertical (axis=1) projection.

    Args:
        R (cv2 image): region to calculate the gradient for
        s (int): smoothing parameter; window to smooth the projection
        axis (int): axis to project
        t (int): maximum number of pixels in a row to consider the row black

    Returns:
        np.ndarray: gradient of the projection

    '''
    ph = np.sum(R > 0, axis)
    ph[ph<t] = 0
    zh = np.zeros_like(ph)
    # s = int(ph.shape[0] * 0.05)
    for x in range(zh.shape[0]):
        i = max(x - s, 0)
        j = min(x + s, zh.shape[0])
        zh[x] = np.floor(np.sum(ph[i:j] / (2*s)))
    if zh.shape[0] < 2:
        return np.array([0])
    gh = np.round(np.gradient(zh, edge_order=1)).astype(np.int)
    
    return gh


def check_homogeneity(R, s: int, axis: int = 1, t: int = 0) -> bool:
    '''Check if a region is homogeneous.

    Using the method outlined in (Tran et al. 2016), calculate the homogeneity
    structure of the region.

    Args:
        R (cv2 image): region to calculate the gradient for
        s (int): smoothing parameter; window to smooth the projection
        axis (int): axis to project
        t (int): maximum number of pixels in a row to consider the row black

    Returns:
        bool: whether the region is homogeneous
    '''
    gh = get_gradient(R, s, axis, t=t)
    lh = [t for t in range(gh.shape[0]-1) if (gh[t] < 0 and gh[t+1] >= 0) or (gh[t] > 0 and gh[t] <= 0)]
    delta = np.array([lh[i+1] - lh[i] for i in range(len(lh)-1)])
    if delta.shape[0] > 0:
        v = np.var(delta)
        return v <= 50
    
    return True


def get_lines(R, axis: int, t: int = 0) -> 'tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]':
    '''Find the black and white lines of a region.

    Use the horizontal or vertical projection to find black lines and white
    lines in the region, respecting the threshold.

    Args:
        R (cv2 image): region to find the lines
        axis (int): axis to project
        t (int): maximum number of pixels in a row to consider the row a white line
    
    Returns:
        tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]: index
        and heights of the white lines and black lines found.
    '''
    p = np.sum(R > 0, axis=axis)

    flags = np.zeros_like(p, dtype=np.bool)
    heights = np.zeros_like(p)
    prev = p[0]

    # flag = True -> black line
    flags = p > t
    heights[0] = 1
    
    for i in range(1, p.shape[0]):
        if (p[i] <= t and prev <= t) or (p[i] > t and prev > t):
            heights[i] = heights[i-1] + 1
        else:
            heights[i] = 1
        
        prev = p[i]
    
    white = []
    black = []
    white_heights = []
    black_heights = []

    bounds = [b for b in np.argwhere(heights == 1).flatten()] + [heights.shape[0]]
    for b in range(len(bounds) - 1):
        start, end = bounds[b], bounds[b+1]
        if flags[start]:
            black.append((end + start) // 2)
            black_heights.append(np.max(heights[start:end]))
        else:
            white.append((end + start) // 2)
            white_heights.append(np.max(heights[start:end]))
        
    return (white, white_heights), (black, black_heights)

def find_last_before(white: 'list[int]', x: int) -> int:
    '''Find the last white line before a certain position.

    Args:
        white (list[int]): list of white lines
        x (int): position
    
    Returns:
        int: the index for the last white line before x, -1 if no white line exists before x
    '''
    k = -1
    for i in range(len(white)):
        if white[i] < x:
            k = i
        else:
            break
    return k

def get_division(R, axis: int, t: int = 0) -> 'list[tuple[int, int]]':
    '''Calculates the positions to divide the region.

    Use the height of black and white lines in the region to calculate the cutting point.

    Args:
        R (cv2 image): region to find the lines
        axis (int): axis to project
        t (int): maximum number of pixels in a row to consider the row a white line

    Returns:
        list[tuple[int, int]]: list of cuts to make along the specified axis
    '''
    (white, white_heights), (black, black_heights) = get_lines(R, axis, t)
    
    wi = np.argwhere((white_heights == np.max(white_heights)) & (white_heights > np.median(white_heights))).flatten() if len(white) > 0 else np.array([])
    bi = np.argwhere((black_heights == np.max(black_heights)) & (black_heights > np.median(black_heights))).flatten() if len(black) > 0 else np.array([])

    div = []
    wdiv = []
    bdiv = []
    if wi.shape[0] > 0: # white division
        prev = 0
        for w in wi:
            wdiv.append((prev, white[w] - white_heights[w] // 2))
            prev = white[w] + white_heights[w] // 2
        wdiv.append((prev, R.shape[1-axis]))
    if bi.shape[0] > 0: # black division
        prev = 0
        for b in bi:
            i = find_last_before(white, black[b])
            if i != -1:
                first = white[i]
                second = white[i+1] if i+1 < len(white) else first
                first = white[b] if b < len(white) else white[-1]
                second = white[b+1] if b+1 < len(white) else white[-1]
                if first == second:
                    bdiv.append((prev, first - white_heights[i] // 2))
                    prev = first + white_heights[i] // 2
                else:
                    bdiv.append((prev, first - white_heights[i] // 2))
                    bdiv.append((first + white_heights[i] // 2, second - white_heights[i+1] // 2))
                    prev = second
        if prev > 0:
            bdiv.append((prev, R.shape[1-axis]))
            
    divs = []
    for d in wdiv + bdiv:
        divs.extend(d)
    divs = sorted(list(set(divs))) # remove duplicates and sort
    divs = [(divs[i], divs[i+1]) for i in range(len(divs)-1)]
    
    return divs


def recursive_splitting(img, rect: np.ndarray, is_text: np.ndarray, area: np.ndarray, t: float = 0.01, do_filter: bool = True) -> 'tuple[list, list[np.ndarray]]':
    '''Split an image into homogeneous regions.

    Use the method describe by (Tran et al. 2016) to split the image into
    multiple homogeneous regions.

    Args:
        img (cv2 image): the image to split
        rect (np.ndarray): bounding box of the all the CCs
        is_text (np.ndarray): boolean mask for the text CCs
        area (np.ndarray): area (number of filled pixels) for each CCs
        t (float): the threshold of pixels to ignore when computing homogeneity
        do_filter (bool): whether to execute the recursive filter when splitting.

    Returns:
        tuple[list, list[np.ndarray]]: list of regions and their coordinates on the original image.
    '''
    finished_regions = []
    finished_coords = []

    regions = [img]
    coords = [(0, 0, img.shape[1], img.shape[0])]

    all_coords = [coords[0]]

    new_regions = [0]
    while len(new_regions) > 0:
        new_regions = []
        new_homo = []
        new_coords = []
        for i in range(len(regions)):
            # print('in', coords[i])
            x, y, w, h = coords[i]
            # s = int(np.sqrt(w*h) * 0.05)

            homo = check_homogeneity(regions[i], int(w*0.05), 0, int(w*t)) and check_homogeneity(regions[i], int(h*0.05), 1, int(h*t))
            if homo:
                # print('homo!')
                finished_regions.append(regions[i])
                finished_coords.append(coords[i])
            else:
                hdivs = get_division(regions[i], 1, int(w * t))
                vdivs = get_division(regions[i], 0, int(h * t))

                divs = []
                for h in hdivs:
                    for v in vdivs:
                        x1, x2 = min(v[0], v[1]), max(v[0], v[1])
                        y1, y2 = min(h[0], h[1]), max(h[0], h[1])
                        divs.append((x1, x2, y1, y2))
                # print('got', len(divs), 'divisions')
                

                for x1,x2,y1,y2 in divs:
                    rct = (x+x1, y+y1, x2-x1, y2-y1)
                    if x2-x1 > 3 and y2-y1 > 3 and rct not in all_coords:
                        # print('found', rct)
                        if do_filter:
                            filtered = regions[i][y1:y2, x1:x2].copy()
                            recursive_filter(filtered, rct, rect, is_text, area)
                            if converge(regions[i][y1:y2, x1:x2], filtered):
                                finished_regions.append(filtered)
                                finished_coords.append(rct)
                            else:
                                new_coords.append(rct)
                                new_regions.append(filtered)
                        else:
                            new_coords.append(rct)
                            new_regions.append(regions[i][y1:y2, x1:x2])
                            # if new_regions[-1].shape[0] != rct[3] or new_regions[-1].shape[1] != rct[2]:
                            #     print(new_regions[-1].shape, rct)
                        all_coords.append(rct)
                if len(divs) == 0:
                    # print('unable to divide')
                    finished_regions.append(regions[i])
                    finished_coords.append(coords[i])
                    
        # print('scanned', len(regions), 'regions.', len(new_regions), 'new regions found')

        regions = new_regions
        homo = new_homo
        coords = new_coords
    
    return finished_regions, finished_coords


### Filtro Recursivo
def converge(region, after_filter) -> bool:
    '''Check the regions against the convergence criteria.

    Args:
        region (cv2 image): region before operation
        after_filter (cv2 image): region after operation

    Returns:
        bool: True if the algorithm converged for this region
    '''
    Su = np.sum(region)
    Sv = np.sum(after_filter)
    return Su == Sv or Sv == 0


def compute_k(omega: np.ndarray) -> float:
    '''Calculate the k-value for each omega list

    Args:
        omega (np.ndarray): array of widths, heights or areas of the CCs in the region
    
    Returns:
        float: the k calculated by the formula defined in (Tran et al. 2016)
    '''
    return max(np.mean(omega) / np.median(omega), np.median(omega) / np.mean(omega))


def compute_suspected_max(omega: np.ndarray, k: float) -> np.ndarray:
    '''Find the suspected non-text elements by the maximum-median filter.

    Args:
        omega (np.ndarray): array of widths, heights or areas of the CCs in the region
        k (float): the k calculated by the formula defined in (Tran et al. 2016)
    
    Returns:
        np.array: boolean mask for the suspected non-text elements
    '''
    return (omega == np.max(omega)) & (omega > k * np.median(omega))


def compute_suspected_min(omega, k):
    '''Find the suspected non-text elements by the minimum-median filter.

    Args:
        omega (np.ndarray): array of widths, heights or areas of the CCs in the region
        k (float): the k calculated by the formula defined in (Tran et al. 2016)
    
    Returns:
        np.array: boolean mask for the suspected non-text elements
    '''
    return (omega == np.min(omega)) & (omega < np.median(omega) / k)


def is_in_range(v: np.ndarray, start: int, end: int) -> np.ndarray:
    '''Checks if the elements in a vector lie in an interval.

    Args:
        v (np.ndarray): vector of CCs to check
        start (int): start of the interval
        end (int): end of the interval
    
    Returns:
        np.ndarray: boolean mask for the CCs that are in the range
    '''
    return (v > start) & (v < end)


def get_neigh(CCu: np.ndarray) -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    '''Find the neighbouring CCs for each CC.

    Use the method described by (Chen et al. 2013) to calculate the neighbours of a CC

    Args:
        CCu (np.ndarray): all the CCs to use in the analysis
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: LNN (left nearest
        neighbour), RNN (right nearest neighbour), LNWS (left nearest white
        space) and RNWS (right nearest whitespace).
    '''
    lnn = np.zeros(CCu.shape[0])
    rnn = np.zeros(CCu.shape[0])
    lnws = np.zeros(CCu.shape[0])
    rnws = np.zeros(CCu.shape[0])
            
    for i in range(CCu.shape[0]):
        CCi = CCu[i]
        overlap1 = is_in_range(CCu[:, 1], CCi[1], CCi[1] + CCi[3])
        overlap2 = is_in_range(CCu[:, 1] + CCu[:, 3], CCi[1], CCi[1] + CCi[3])
        overlap3 = is_in_range(CCi[1], CCu[:, 1], CCu[:, 1] + CCu[:, 3])
        overlap4 = is_in_range(CCi[1] + CCi[3], CCu[:, 1], CCu[:, 1] + CCu[:, 3])
        vertical_overlap = overlap1 | overlap2 | overlap3 | overlap4
        ws_left = (CCu[:,0] + CCu[:,2]) - CCi[0]
        ws_right = CCu[:,0] - (CCi[0] + CCi[2])
        _lnn = np.argsort(ws_left)
        _rnn = np.argsort(ws_right)
        _lnn = _lnn[vertical_overlap[_lnn]]
        _rnn = _rnn[vertical_overlap[_rnn]]
        lnn[i] = _lnn[0] if _lnn.shape[0] > 0 else -1
        rnn[i] = _rnn[0] if _rnn.shape[0] > 0 else -1
        lnws[i] = ws_left[_lnn[0]] if _lnn.shape[0] > 0 else -1
        rnws[i] = ws_right[_rnn[0]] if _rnn.shape[0] > 0 else -1
    
    return lnn, rnn, lnws, rnws


def get_cc_in_region(region: np.ndarray, cc: np.ndarray) -> 'list[tuple[int, int, int, int]]':
    '''Find all the CCs contained in a region.

    Args:
        region (np.ndarray): bounding box of the region
        cc (np.ndarray): bounding box for all of the CCs in the image
    
    Returns:
        list[tuple[int, int, int, int]]: list of the bounding boxes of all the
        CCs contained in the region.
    '''
    return [(cc[i][0],cc[i][1],cc[i][2],cc[i][3], i) for i in range(cc.shape[0]) if cc[i][0] > region[0] and cc[i][0]+cc[i][2] < region[0]+region[2] and cc[i][1] > region[1] and cc[i][1]+cc[i][3] < region[1]+region[3]]


def recursive_filter(region, coords: np.ndarray, rect: np.ndarray, is_text: np.ndarray, area: np.ndarray):
    '''Apply the recursive filter to a region.

    Use the recursive filter described by (Tran et al. 2016) to eliminate
    non-text elements not caught by the heuristic filter.

    Args:
        region (cv2 image): image to apply the filter
        coords (np.ndarray): bounding box of the region
        rect (np.ndarray): bounding box of the all the CCs
        is_text (np.ndarray): boolean mask for the text CCs
        area (np.ndarray): area (number of filled pixels) for each CCs
    '''
    CCs = np.array(get_cc_in_region(coords, rect[is_text]))
    if CCs.shape[0] == 0: return
    indicies = CCs[:,-1]
    CCu = CCs[:,:-1]

    omega1 = area[is_text][indicies]#np.array([CCi[2]*CCi[3] for CCi in CCu])
    omega2 = np.array([CCi[3] for CCi in CCu])
    omega3 = np.array([CCi[2] for CCi in CCu])
    
    lnn, rnn, lnws, rnws = get_neigh(CCu)
    
    num_ln = np.array([(lnn == i).sum() for i in range(lnn.shape[0])])
    num_rn = np.array([(rnn == i).sum() for i in range(rnn.shape[0])])
    ws = rnws[rnws > 0] if (rnws>0).any() else np.array([0])

    k1, k2, k3 = compute_k(omega1), compute_k(omega2), compute_k(omega3)

    # maximum median filter

    suspected = compute_suspected_max(omega1, k1) & (compute_suspected_max(omega2, k2) | compute_suspected_max(omega2, k3))

    lnws[lnws == -1] = 1e10
    rnws[rnws == -1] = 1e10
    mi = np.min([lnws, rnws], axis=0)
    cond1 = mi > max(np.median(ws), np.mean(ws))

    lnws[lnws == 1e10] = -1
    rnws[rnws == 1e10] = -1
    ma = np.max([lnws, rnws], axis=0)
    cond1 &= (ma == np.max(ws)) | (mi > 2 * np.mean(ws))

    cond2 = (num_ln == np.max(num_ln)) & (num_ln > 2)
    cond2 |= (num_rn == np.max(num_rn)) & (num_rn > 2)

    non_text = suspected & (cond1 | cond2)

    # minimum median filter

    suspected = compute_suspected_min(omega2, k2) | compute_suspected_min(omega3, k3)

    lnws[lnws == -1] = 1e10
    rnws[rnws == -1] = 1e10
    mi = np.min([lnws, rnws], axis=0)
    cond1 = mi > max(np.median(ws), np.mean(ws))

    non_text |= suspected & cond1

    i = 0
    for x,y,w,h in CCu[non_text]:
        x -= coords[0]
        y -= coords[1]
        cv2.rectangle(region, (x, y), (x+w, y+h), 0, -1)
        is_text[is_text][indicies[i]] = False
        i += 1


### Classificação Multi-Layer
def multi_layer(img, rect: np.ndarray, is_text: np.ndarray, area: np.ndarray, t: float = 0):
    '''Apply the multy-layer classification to an image.

    Use the method described by (Tran et al. 2017) to eliminate further non-text
    elements.

    Args:
        img (cv2 image): image to apply the ML classification
        rect (np.ndarray): bounding box of the all the CCs
        is_text (np.ndarray): boolean mask for the text CCs
        area (np.ndarray): area (number of filled pixels) for each CCs
        t (float): the threshold of pixels to ignore
    
    Returns:
        cv2 image: text image after the removal of all the non-text elements
    '''
    prev = img.copy() * 0
    current = img.copy()
    i = 0
    while not converge(prev, current):
        rs = []
        cs = []
        hdivs = get_division(current, 1, int(img.shape[0] * t))
        vdivs = get_division(current, 0, int(img.shape[1] * t))
        divs = []
        for h in hdivs:
            for v in vdivs:
                x1, x2 = min(v[0], v[1]), max(v[0], v[1])
                y1, y2 = min(h[0], h[1]), max(h[0], h[1])
                divs.append((x1, x2, y1, y2))        

        for x1,x2,y1,y2 in divs:
            rct = (x1, y1, x2-x1, y2-y1)
            cs.append(rct)
            rs.append(current[y1:y2, x1:x2])
        
        prev = current
        current = current.copy() * 0
        for i in range(len(rs)):
            recursive_filter(rs[i], cs[i], rect, is_text, area)
            x,y,w,h = cs[i]
            current[y:y+h, x:x+w] = rs[i]
        i += 1
    # print(i, 'iterations')
    return current


def segment(img_bw, temp_folder: str = None, output_path: str = None) -> 'tuple[np.ndarray, list, list[np.ndarray]]':
    '''Segment an image using an MHS based approach.

    Implements a MHS (Tran et al. 2017) based approach for document text region
    identification based on homogeneity.

    Args:
        img_bw (cv2 image): binarized image to segment
        temp_folder (str): folder to save intermediary files to, if None does not save. default=None
        output_path (str): path to the resulting image with only text elements, if None does not save. default=None
    
    Returns:
        tuple[np.ndarray, list, list[np.ndarray]]: the text document, a list of
        all the regions and all of their coordinates.
    '''

    _, thresh = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    area, density, rect, inc, hw_rate = cc_analisys(thresh)

    thresh, is_text = heuristic_filter(thresh, area, density, rect, inc, hw_rate)
    conditional_save(thresh, get_conditional_path('heuristic_filter.png', temp_folder))

    # in case there is a text element that is now empty, make it non-text
    for i in range(rect.shape[0]):
        if is_text[i]:
            x,y,w,h = rect[i]
            is_text[i] = np.any(thresh[y:y+h,x:x+w] > 0)
    
    if temp_folder:
        img_boxes = thresh.copy()
        for r in rect[is_text]:
            x,y,w,h = r
            cv2.rectangle(img_boxes, (x,y), (x+w,y+h), 128, 2)
        conditional_save(img_boxes, get_conditional_path('text_ccs.png', temp_folder))
    
    # print('before:', is_text.sum())
    rs, cs = recursive_splitting(thresh, rect, is_text, area, t=0.01)
    # print('after:', is_text.sum())
    
    # remove empty(-ish) regions
    new_rs = [rs[i] for i in range(len(rs)) if np.sum(rs[i] > 0) / (cs[i][2]*cs[i][3]) > 0.01]
    new_cs = [cs[i] for i in range(len(rs)) if np.sum(rs[i] > 0) / (cs[i][2]*cs[i][3]) > 0.01]
    
    rs, cs = new_rs, new_cs

    if temp_folder:
        img_boxes = thresh.copy()
        for r in cs:
            x,y,w,h = r
            cv2.rectangle(img_boxes, (x,y), (x+w,y+h), 128, 2)
        conditional_save(img_boxes, get_conditional_path('multilevel_regions.png', temp_folder))
        
    img = thresh.copy() * 0
    for i in range(len(rs)):
        x,y,w,h = cs[i]
        img[y:y+h, x:x+w] = rs[i]
    conditional_save(img_boxes, get_conditional_path('multi_level.png', temp_folder))
    
    # remove the text CCs now empty
    CCt = np.argwhere(is_text).flatten()
    for i in CCt:
        x,y,w,h = rect[i]
        if np.sum(img[y:y+h, x:x+w] > 0) == 0:
            is_text[i] = False


    # print('before:', is_text.sum())
    img = multi_layer(img, rect, is_text, area, t=0.01)
    # print('after:', is_text.sum())
    conditional_save(img, get_conditional_path('multi_layer.png', temp_folder))
    
    ### Segmentação de Regiões Homogêneas
    rs, cs = recursive_splitting(img, rect, is_text, area, t=0, do_filter=False)
    new_rs = [rs[i] for i in range(len(rs)) if np.sum(rs[i] > 0) / (cs[i][2]*cs[i][3]) > 0.01]
    new_cs = [cs[i] for i in range(len(rs)) if np.sum(rs[i] > 0) / (cs[i][2]*cs[i][3]) > 0.01]
    rs, cs = new_rs, new_cs

    if temp_folder:
        img_boxes = img.copy()
        for r in cs:
            x,y,w,h = r
            cv2.rectangle(img_boxes, (x,y), (x+w,y+h), 128, 2)
        conditional_save(img_boxes, get_conditional_path('mhs_boxes.png', temp_folder))
    
    conditional_save(img, output_path)
        
    return img, rs, cs
