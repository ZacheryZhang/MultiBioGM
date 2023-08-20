import os
import shutil
import glob
from tqdm import tqdm
name="modify"
toname="origin5"
if not os.path.exists(toname):
	os.mkdir(toname)

if name=="Union_db1_traingt1":
	path1s="Union_db1/train_GT/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_testgt1":
	path1s="Union_db1/test_GT/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_traingt2":
	path1s="Union_db1/train_GT/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_testgt2":
	path1s="Union_db1/test_GT/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_traingt3":
	path1s="Union_db1/train_GT/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_testgt3":
	path1s="Union_db1/test_GT/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)

if name=="Union_db1_train1":
	path1s="Union_db1/train/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_test1":
	path1s="Union_db1/test/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_train2":
	path1s="Union_db1/train/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_test2":
	path1s="Union_db1/test/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_train3":
	path1s="Union_db1/train/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db1_test3":
	path1s="Union_db1/test/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)

if name=="Union_db2_train1":
	path1s="Union_db2/train/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_test1":
	path1s="Union_db2/test/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_train2":
	path1s="Union_db2/train/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_test2":
	path1s="Union_db2/test/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_train3":
	path1s="Union_db2/train/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_test3":
	path1s="Union_db2/test/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)

if name=="Union_db2_1":
	path1s="Union_db2/result/AttU_Net/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_2":
	path1s="Union_db2/result/AttU_Net/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db2_3":
	path1s="Union_db2/result/AttU_Net/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)

if name=="Union_db3_1":
	path1s="Union_db3/test/fingerprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingerprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db3_2":
	path1s="Union_db3/test/fingervein/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"fingervein_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="Union_db3_3":
	path1s="Union_db3/test/palmprint/"
	for path1 in os.listdir(path1s):
		if path1==".DS_Store":
			continue
		base=os.path.join(path1s,path1)
		for img in os.listdir(base):
			imgpath=os.path.join(base,img)
			topath=os.path.join(toname,"palmprint_"+path1+"_"+img)
			shutil.copy(imgpath,topath)
if name=="modify":
	paths=glob.glob(f"{toname}/*.png")
	from PIL import Image
	for path in tqdm(paths):
		img=Image.open(path)
		new_img=img.resize((224,224),Image.BILINEAR)
		new_img.save(path)
