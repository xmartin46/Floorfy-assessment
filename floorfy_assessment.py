import cv2
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt

def findImageInsideEquirectangular(equirectImgPath, imgPath, methodName='cv2.TM_CCOEFF', showProcess=False):
    # Load images in greyscale
    equirectImgOrig = cv2.imread(equirectImgPath, 0)
    imgOrig = cv2.imread(imgPath, 0)

    if showProcess:
        cv2.imshow("Equirectangular image", cv2.resize(equirectImgOrig, (720, 720)))
        cv2.imshow("Sub image", cv2.resize(imgOrig, (720, 720)))
        cv2.waitKey()
    
    equirectImg = cv2.Canny(equirectImgOrig, 150, 220)
    img = cv2.Canny(imgOrig, 150, 220)

    if showProcess:
        cv2.imshow("Equirectangular image", cv2.resize(equirectImg, (720, 720)))
        cv2.imshow("Sub image", cv2.resize(img, (720, 720)))
        cv2.waitKey()

    width, height = img.shape[::-1]

    # Apply template Matching
    method = eval(methodName)
    res = cv2.matchTemplate(equirectImg, img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Show image and rectangle
    rect = patches.Rectangle(top_left, width, height, linewidth=1, edgecolor='r', facecolor='None')
    
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # using tuple unpacking for multiple Axes
    ax1.imshow(cv2.cvtColor(cv2.imread(equirectImgPath), cv2.COLOR_BGR2RGB))
    ax1.set_title('Template matching result')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.add_patch(rect)
    
    ax2.imshow(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB))
    ax2.set_title('Sub-Image')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    fig.suptitle(f"Method used: {methodName}")

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

    showProcess = False

    for subImgPath in subImgPaths:
        top_left, bottom_right = findImageInsideEquirectangular(equirectImgPath, subImgPath, methodName=methods[0], showProcess=showProcess)
        print(f"Sub-Image '{subImgPath}' has been found in the rectangle starting from pixel top-left {top_left} to pixel bottom-right {bottom_right}.")

if __name__=="__main__":
    main()