from tqdm import tqdm
import cv2
import math
import glob
import pandas as pd
import numpy as np

def sampleToFolder(inpDir, outDir):
	fileList1 = [f for f in glob.glob(inpDir + "*.avi")]
	for i in tqdm(range(len(fileList1))):
	    count = 0
	    videoFile = fileList1[i]
	    cap = cv2.VideoCapture(videoFile)   
	    frameRate = cap.get(5)
	    x=1
	    while(cap.isOpened()):
	        frameId = cap.get(1) #current frame number
	        ret, frame = cap.read()
	        if (ret != True):
	            break
	        if (frameId % math.floor(frameRate) == 0):
	            # storing the frames in a new folder named train_1
	            filename = "outDir" + videoFile.split('/')[-1].split('.')[0] +"_frame%d.jpg" % count;count+=1
	            # print(filename)
	            cv2.imwrite(filename, frame)
	    cap.release()

def dataFromFolder(inpDir):
	picDir = inpDir + "*.jpg"
	images = glob.glob(picDir)
	imageName = []
	imageClass = []
	for i in tqdm(range(len(images))):
	  str = images[i].split('/')[-1]
	  imageName.append(str)
	  imC = 2 #Yawning
	  if str.find('Normal') != -1:
	    imC = 0
	  elif str.find('Talking') != -1:
	    imC = 1
	  imageClass.append(imC)
	return imageName, imageClass

def createDatasetFolder(dFrame, inpDir):
  data = dFrame
	imageData = []
	tDr  = inpDir
	for i in tqdm(range(data.shape[0])):
	    
	    img = image.load_img(tDr+data['image'][i], target_size=(224,224,3))
	    img = image.img_to_array(img)
	    img = img/255
	    imageData.append(img) 
	return imageData

# dataset for RNN models
def createSeqDataset(inpDir):
	fileList1 = [f for f in glob.glob(inpDir + "*.avi")]
	classList = []
	vidList = []
	fNameList = []
	for i in tqdm(range(len(fileList1))):

	  count = 0
	  temp = []
	  videoFile = fileList1[i]
	  cap = cv2.VideoCapture(videoFile)   
	  frameRate = cap.get(5)

	  while(cap.isOpened()):
	    frameId = cap.get(1) #current frame number
	    ret, frame = cap.read()
	    if (ret != True):
	      break
	    if (frameId % math.floor(frameRate/2) == 0): # 2fps
	      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	      frame = Image.fromarray(frame)

	      # resize the array (image) then PIL image
	      frame = frame.resize((128, 128))
	      frame = image.img_to_array(frame)
	      frame = frame/255
	      temp.append(frame)
	  cap.release()
	  vidList.append(temp)
	  str1 = videoFile.split('/')[-1]
	  fNameList.append(str1)
	  str1 = str1.split('.')[0].split('-')[-1]
	  imC = 2 #Yawning
	  if str1 == 'Normal':
	    imC = 0
	  elif str1 == 'Talking':
	    imC = 1
	  classList.append(imC)
	return vidList, classList, fNameList

def createMultiFrameDataset(seqData, dFrame):
	X = seqData
	data = dFrame
	y = data['class']
	fName = data['image']
	casList = []
	nClsList = []
	nFL = []
	for l in tqdm(range(0, len(X))):
	  row = X[l]
	  k  = math.floor(len(row)/6)
	  for i in range(0,k):
	    temp = np.hstack([row[j*k + i] for j in range(0,6)])
	    casList.append(temp)
	    nClsList.append(y[l])
	    nFL.append(fName[l])
	return casList, nClsList, nFL