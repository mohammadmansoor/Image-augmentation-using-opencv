import cv2
import os
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float

in_cwd = os.getcwd() + "/input/"
out_cwd = os.getcwd() + "/output/"

in_img = os.listdir(in_cwd)


class augmentation:
   
    def rotation(self, image, degree,img):
        h, w, c = image.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
        image = cv2.warpAffine(image, M, (w, h))
        cv2.imwrite(out_cwd +"Rotation-" +str(degree)+img, image)

    def flip(self,image, dir,img):
        image = cv2.flip(image, dir)
        cv2.imwrite(out_cwd +"Flip-" +str(dir)+ img, image)

    def resizing(self,image,w,h,img):
        image=cv2.resize(image,(w,h))
        cv2.imwrite(out_cwd+"Resize-"+str(w)+"*"+str(h)+img, image)

    def grayscale(self,image,img):
        image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_cwd + "Grayscale-" + img, image)

    def cropping(self,image,rt,lt,tp,bt,img): 
        image=image[bt:tp,lt:rt]
        cv2.imwrite(out_cwd+"Cropping-"+str(lt)+str(rt)+"*"+str(bt)+str(tp)+img, image)

    def sharpe(self,image,img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(out_cwd+"Sharpe-"+img, image)

        
aug = augmentation()

for img in in_img:

    image = cv2.imread(in_cwd + img)

    aug.rotation(image, 90,img)
    aug.rotation(image, 180,img)
    aug.rotation(image, 270,img)

    aug.flip(image, 1,img)
    aug.flip(image, 0,img)

    aug.resizing(image,400,400,img)
    aug.resizing(image,300,300,img)

    aug.grayscale(image,img)

    aug.cropping(image,100,400,0,350,img)
    aug.cropping(image,400,100,300,100,img)

    aug.sharpe(image,img)
    
