import matplotlib.pyplot as plt
import cv2
import numpy as np

original_image = cv2.imread("aayushi.jpg")
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
vector = img.reshape((-1,3))
vector = np.float32(vector)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
attempts=10
ret,label,center=cv2.kmeans(vector,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()


color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([result_image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
#hist = cv2.calcHist([result_image], [0, 1, 2], None, [256],[0,256])