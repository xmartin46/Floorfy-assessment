import cv2
import imutils
import numpy as np
import matplotlib.patches as patches
from tqdm import tqdm
from matplotlib import pyplot as plt

def findImageInsideEquirectangular(equirectImgPath, imgPath, minRange = 0.5, maxRange = 2.0, spaces = 30, methodName='cv2.TM_CCOEFF'):
    equirectImgOrig = cv2.imread(equirectImgPath)

    equirectImg = cv2.imread(equirectImgPath, 0)
    img = cv2.imread(imgPath, 0)

    equiWidth, equiHeight = equirectImg.shape[::-1]
    imgWidth, imgHeight = img.shape[::-1]

    img = cv2.Canny(img, 150, 220)

    found = None

    method = eval(methodName)

    for scale in tqdm(np.linspace(minRange, maxRange, spaces)[::-1]):
        resizedImg = imutils.resize(equirectImg, width = int(equiWidth * scale))
        resizedWidth, resizedHeight = resizedImg.shape[::-1]

        ratio = equiWidth / float(resizedWidth)

        # if equirectImg is smaller than img, end loop
        if resizedWidth < imgWidth or resizedHeight < imgHeight:
            break

        edgedImg = cv2.Canny(resizedImg, 50, 220)
        result = cv2.matchTemplate(edgedImg, img, method)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                found = (minVal, minLoc, ratio)
            else:
                found = (maxVal, maxLoc, ratio)

    (_, maxLoc, ratio) = found
    top_left = tuple((int(maxLoc[0] * ratio), int(maxLoc[1] * ratio)))
    bottom_right = tuple((int(top_left[0] + imgWidth * ratio), int(top_left[1] + imgHeight * ratio)))
    rect = patches.Rectangle(top_left, imgWidth * ratio, imgHeight * ratio, linewidth=1, edgecolor='r', facecolor='None')
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(cv2.imread(equirectImgPath), cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Template matching result (ratio = {ratio})")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.add_patch(rect)
    
    ax2.imshow(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB))
    ax2.set_title('Sub-Image')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    fig.suptitle(f"MULTI-SCALED TEMPLATE MATCHING \n Method used: {methodName}")

    plt.show()

    return top_left, bottom_right

def main():
    equirectImgPath = 'equirectangular_salon.jpg'
    subImgPaths = ['Captura de pantalla 2021-03-03 a las 13.59.06.png',
                    'Captura de pantalla 2021-03-03 a las 14.00.02.png',
                    'Captura de pantalla 2021-03-03 a las 14.00.09.png']

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
                'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
                'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for subImgPath in subImgPaths:
        top_left, bottom_right = findImageInsideEquirectangular(equirectImgPath, subImgPath, methodName=methods[0])
        print(f"Sub-Image '{subImgPath}' has been found in the rectangle starting from pixel top-left {top_left} to pixel bottom-right {bottom_right}.")

if __name__=="__main__":
    main()