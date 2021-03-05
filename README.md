# FLOORFY ASSIGNMENT

## Libraries used
* cv2
* imutils
* numpy
* matplotlib
* tqdm

## How to run
```Python
git clone https://github.com/xmartin46/Floorfy-assessment.git
```

```Python
python3 <file-name>
```

## Version 1 (floorfy_assessment.py)
The first idea I had focused on Template Matching. First using canny edge detection for later using cv2.matchingTemplate(). By doing so, we achieve to know the location of the object. However, as the size is different, the rectangle created is not well fitted.

## Version 2 (floorfy_assessment_multiscaleTemplateMatching.py)
In order to solve the problem in the first version, I have varied the size of the equirectangular image from 2.0 to 0.5 scale. With this, we achieve the best possible solution in any of the 3 sub-images. However, it is slow (although we can parallelize it).

## Version 3 (floorfy_assessment_SIFT-Homography.py)
In this approach, I have tried to obtain the features and descriptors of both images for later matching them. This version is much faster than version 2. However, for the image 'Captura de pantalla 2021-03-03 a las 14.00.02.png', it does not obtain any descriptors and therefore we get only 2/3 sub-images correctly.

## Version 4 (floorfy_assessment_merged.py)
This final solution tries to merge version 2 and 3 approaches to solve both problems (speed and accuracy). To do so, we first use version 3 approach and, if it does not obtain any descriptors for the sub-image, we then use version 2 algorithm to solve it. This solution, as well as the second, achieves the best possible solution in any of the 3 sub-images. Moreover, it is faster because it first tries to use version 3.
