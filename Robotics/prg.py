import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

image=input("Hi there,Welcome to the image recognition server!")
img = cv2.imread(image)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img1)
plt.show()
box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)
output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(output)
plt.show()
print("Number of objects in this image are " +str(len(label)))
res={}
for key in label:
	res[key]=label.count(key)

print(res)