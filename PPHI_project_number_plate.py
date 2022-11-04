import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('/content/download (1).png')
img = cv2.resize(img, (800, 600))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting image to grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blur, 10, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

location = None
for contour in contours:
    contourPerimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02*contourPerimeter, True)
    # print(approx)
    if len(approx) == 4:
        location = approx
        break

location

(x, y, w, h) = cv2.boundingRect(location)
license_plate = gray[y:y + h, x:x + w]

license_plate

mask = np.zeros(edged.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x, y) = np.where(mask == 255)  # Finding all points where our image is not black
# he minimum point will be the top left corner of the plate
(x1, y1) = (np.min(x), np.min(y))
# The max points will be bottom right corner of the plate
(x2, y2) = (np.max(x), np.max(y))
# Added the +1 to leave some room for error
cropped_plate = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_plate)
result

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60),
                  fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(
    approx[2][0]), (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
