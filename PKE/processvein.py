import cv2
import os
import shutil
def rotate(img):
	img=cv2.transpose(img)
	img=cv2.flip(img,1)
	return img
base="./Union_db1/train/fingerprint"
clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(5, 5))
files=os.listdir(base)
for i in files:
	name=i
	for j in os.listdir(os.path.join(base,i)):
		p=os.path.join(base,i,j)
		#p1=os.path.join(base,name,"%d.png"%2)
		#shutil.copy(p,p1)
		img=cv2.imread(p,0)
		#img=rotate(img)
		img=clahe.apply(img)
		cv2.imwrite(p,img)
		
