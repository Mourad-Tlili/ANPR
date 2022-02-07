# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
from webcolors import rgb_to_name,CSS3_HEX_TO_NAMES,hex_to_rgb
from scipy.spatial import KDTree
from io import BytesIO
import pytesseract
import requests
import torchvision
from PIL import Image
import numpy as np
from torchvision import transforms
import os.path
import time
import mysql.connector
import datetime
from pytesseract import image_to_string
import argparse
from PIL import Image, ImageEnhance, ImageFilter
from mysql.connector import connect, Error 
from imutils import paths
import argparse
import imutils
import skimage

connection_params = {
	'host': "localhost",
	'user': "root",
	'password': "",
	'database': "employeesystem",
}

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

ap = argparse.ArgumentParser()
ap.add_argument('--i', '-input', required=True,help="path to input directory of images", type=str)
#parser.add_argument('--video', help='Path to video file.')
args = ap.parse_args()
imagePaths = sorted(list(paths.list_images(args.i)))
s=0
for imagePath in imagePaths:

# Load names of classes
	classesFile = "classes.names";

	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Give the configuration and weight files for the model and load the network using them.

	modelConfiguration = "darknet-yolov3.cfg";
	modelWeights = "lapi.weights";

	net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
	net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

	# Get the names of the output layers
	def getOutputsNames(net):
		# Get the names of all the layers in the network
		layersNames = net.getLayerNames()
		# Get the names of the output layers, i.e. the layers with unconnected outputs
		return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# Draw the predicted bounding box
	def drawPred(classId, conf, left, top, right, bottom):
		# Draw a bounding box.
		#    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
		# serie ligne
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		
		Cropped=frame[top:bottom, left:right]
		Cropped = cv.resize(Cropped,(200,50))
	
		cv.imshow('test 1',Cropped)
		gray = cv.cvtColor(Cropped, cv.COLOR_BGR2GRAY) 
	# gray = cv.bilateralFilter(gray, 13, 15, 15) 
	# retval, dst = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
	# image = np.hstack((gray, dst))
		#blurImg = cv.blur(image,(250,150)) 
	# imgGray = color.rgb2gray(img)
		
		im = Image.fromarray(Cropped)  # img is the path of the image
		im = im.convert("RGBA")
		newimdata = []
		datas = im.getdata()

		for item in datas:
			if item[0] < 112 or item[1] < 112 or item[2] < 112:
				newimdata.append(item)
			else:
				newimdata.append((255, 255, 255))
		im.putdata(newimdata)

		im = im.filter(ImageFilter.MedianFilter())
		enhancer = ImageEnhance.Contrast(im)
		im = enhancer.enhance(2)
		im = im.convert('1')
		im.save('temp2.jpg')
		
		cv.imwrite("filename3.jpg", Cropped)
		image = Image.open('filename3.jpg')
	# print("size:")
	# print(image.size)
		contrast = ImageEnhance.Contrast(image)
		contrast.enhance(1.5).save('contrast.jpg')
		image2 = Image.open('contrast.jpg') 
		#print("size contrast:")
		#print(image2.size)
		from ArabicOcr import arabicocr
		img5 = cv.imread('filename3.jpg')
		img5 = cv.cvtColor(img5, cv.COLOR_RGB2GRAY) 
		roi = cv.threshold(img5, 0, 255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
		cv.imwrite("resultat6.jpg", roi)
		_, result4 = cv.threshold(roi, 70, 255, cv.THRESH_BINARY_INV)
		cv.imwrite("resultat7.jpg", result4)
		
		#imgGray = color.rgb2gray(imagePath)
		#cv.imwrite("imgGray.jpg", imgGray)
		img = Image.open(imagePath).convert("RGBA")
		img.save('greyscale.png')
		image_path="greyscale.png"
		out_image='out.jpg'
		results=arabicocr.arabic_ocr(image_path,out_image)
		print(results)
		words=[]
		tab1=['ك','ع','>','T']
		for i in range(len(results)):
			
			word=results[i][1]
			StrA = "".join(word)	
			print("word="+word)
			if len(word)>=9 and word.find('ن')!=-1 and word.find('ن')!=len(word)-1 or len(word)>=9 and word.find('و')!=-1 and word.find('و')!=len(word)-1 or len(word)>=9 and word.find('`')!=-1 and word.find('`')!=len(word)-1:
				words.append(word)
			elif words==[] and StrA.isnumeric()== True  and len(StrA)<=4 or words!=[] and len(words[0])<=4 and StrA.isnumeric()== True and len(StrA)<=4:
				words.append(word)
			
			max_index = len(word)
		for i in range(len(words)):
			if words!=[] and len(words)==2 and len(words[1])>=9:
				words.remove(words[0])

		
 
		print("words=")
		print(words)
		if words==[]:
			r1=pytesseract.image_to_string(Cropped, config='-l ara --oem 1 --psm 3')
			StrA = "".join(r1)
			StrA=StrA.replace('_','').replace('|','')
			newstr = StrA.strip()
			words.append(newstr)
		print("words=")
		print(words)	
		print(len(StrA))
		max_index = len(word)
		print("max_index ="+str(max_index))
		pl=[]
		for i in range(len(words)):
			word=results[i][1]
			StrA = "".join(word)
			#pl.append(word)
			
			if len(words)==1 and len(word)==max_index and max_index>=13 and word.find('T')==-1 and word.find('S')==-1 and word.find('A')==-1 and word.find('ك')==-1:
				word.replace(':','').replace('/','').replace('|','').replace(';','').replace('٥','0')
				pl.append(word)
			if len(words)==3 or len(words)==2 and StrA.isnumeric()=="True":
				pl.append(word)	
				
		print("pl==")
		print(pl)
		with open ('file.txt','w',encoding='utf-8')as myfile:
			myfile.write(str(words))
		import cv2
		img = cv2.imread('out.jpg', cv2.IMREAD_UNCHANGED)
		cv2.imshow("arabic ocr",img)
		if len(words)==1:
				k = "".join(words)
				k=k.replace('.','').replace('|','').replace('/','4').replace(']','').replace(';','').replace(':','').replace('٥','0').replace('[','')
				pm1=pytesseract.image_to_string(image_path, config='-l ara --oem 1 --psm 3')
				if k.find("و") != -1 and k.find("و") != len(k)-1 or k.find("ن") != -1 and k.find("ن") != len(k)-1 or k.find("`") != -1 and k.find("`") != len(k)-1 or k.find("ض") != -1 and k.find("ض") != len(k)-1:
					print("matricule1 =")
					if k.find("L") == -1:
						if k[len(k)-3]!=1 and k[len(k)-3]!=2 and k[len(k)-3]!=3 and k[len(k)-3]!=4 and k[len(k)-3]!=5 and k[len(k)-3]!=6 and k[len(k)-3]!=7  and k[len(k)-3]!=8 and k[len(k)-3]!=9:
							k=k.replace(k[len(k)-3],'1')
						p=k[len(k)-3:]+"TN"+k[0:4]
						if p[0]=='ن':
							p=p.replace(p[0],'1')
						print(p)
					else:
						#image_path1= image2.name+'.jpg'
						results1=arabicocr.arabic_ocr(image_path,out_image)
						print(k[len(k)-3:]+"TN"+k[0:4])
						#print(k[len(k)-3:]+"TN"+k[0:4])
						#print(results1)
				if k[len(k)-3:].isnumeric()==False:
					text3 = pytesseract.image_to_string(Cropped, config='-l eng --psm 11')
					text4 = pytesseract.image_to_string(Cropped, config='-l fra --oem 1 --psm 3')
					text5 = pytesseract.image_to_string(Cropped, config='-l fao --oem 1 --psm 1')
					text6 = pytesseract.image_to_string(Cropped, config='-l ara --oem 1 --psm 3')
					text3 = text3.replace('[','').replace(']','')
					kk=text3[len(text3)-6:] 
					p=text3[0:3]
					print("matricule")
					if p in text4 or p in text3 or p in text6 or p  in text5 and int(p) < 200:
						tt=text3[0:3] 
					if text4[0]=='|' or text3[0]=='|' or text33[0]=='|':
						tt=tt.replace(tt[0],'')  
					else:
						tt=text3[0:2]
					if L !=[] or len(text6)==5:
						result=tt+"TN"+kk
					elif kk.find(" ") !=0:
						result=tt+"TN"+kk[kk.find(" ")+1:]
					else:
						result=tt+"TN"+kk	
					print("matricule ="+ result)

						
				
		if len(words)==2:
						

				
			k1 = "".join(words[0])
			k2 = "".join(words[1])
			pm2=pytesseract.image_to_string(imagePath, config='-l ara --oem 1 --psm 3')
			print("matricule =")
			print(k1[0:4]+"TN"+k2[0:4])

	
		    
		if len(words)==3:
			k1 = "".join(words[0])
			k2 = "".join(words[2])
			k3 = "".join(words[1])
			if k3.find("ت") !=-1 and k3.find("و") != -1 or k3.find("ن") != -1 and k2 !="":
				print("matricule3 =")
				print(k1+"TN"+k2)
			else:
				print("matricule3 =")
				print(k1+"TN"+k3)	
             
		text33 = pytesseract.image_to_string(Cropped, config='-l eng --oem 1 --psm 3')
		text3 = pytesseract.image_to_string(Cropped, config='-l eng --psm 11')
		text4 = pytesseract.image_to_string(Cropped, config='-l fra --oem 1 --psm 3')
		text44 = pytesseract.image_to_string(Cropped, config='-l fra --oem 1 --psm 4')
		text5 = pytesseract.image_to_string(Cropped, config='-l fao --oem 1 --psm 3')
		text5 = pytesseract.image_to_string(Cropped, config='-l fao --oem 1 --psm 1')
		text6 = pytesseract.image_to_string(Cropped, config='-l ara --oem 1 --psm 3')
		
		text3=text3.replace("| ","").replace(" |","").replace("°","").replace("}","").replace("[","").replace("L","").replace("t ","").replace(":","").replace("M","").replace("]","").replace(".","").replace("(","").replace('"','').replace(')','').replace(';','').replace('*','').replace('“','')
		#print("position")
		#print(text3.find("-"))
		#print("length")
		#print(len(text3))
		#print("length arabe")
		#print(len(text6))
		tab=['ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص','ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي']
		tab1=['a','b','c','d','e','f','j','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','z']
		L=[]
		for mot in tab:
			if mot in text6:
			    L.append(text6.find(mot))
		if L!=[]:
			print("taille de L")
			#print(len(L))     
		if text3.find("-") == len(text3)-3:
			text3=text3.replace(text3[len(text3)-3],"")
			#print("final :"+text3)
		k=text3[len(text3)-6:]    
		#print("k= "+k)
		text4=text4.replace(" ","").replace(":","").replace("]","")
		text5=text5.replace(" ","").replace(":","").replace("]","").replace("[","").replace("|","").replace(".","")
		#text3=text5.replace(" ","").replace(":","").replace("]","").replace("[","").replace("|","").replace(".","")
		#text33=text5.replace(" ","").replace(":","").replace("]","").replace("[","").replace("|","").replace(".","")
		text6=text6.replace(" ","").replace(":","").replace("]","")
		#print("text33 "+text33)
		#print("text3 "+text3)
		#print("partie de arabe:"+text6[0:4])
		aa=text6.replace("وس","")
		#print("aa "+aa)
		#print("k "+k)
		#print(text4[0:3])
		kk=text3[len(text3)-6:] 
		p=text3[0:3]
		p1=text3[1:4]
		print("p1 :"+p1)
		#print("text4= "+text4)
		#print("text3= "+text3)        
		if p in text4 or p in text3 or p in text6 or p  in text5 and int(p) < 200:
			tt=text3[0:3] 
			if text4[0]=='|' or text3[0]=='|' or text33[0]=='|':
				tt=tt.replace(tt[0],'')  
		else:
			tt=text3[0:2]    
		#print("***text3*** "+text3)
		#print("***text33*** "+text33)
		#print("**text44**** "+text44)
		#print("**text4**** "+text4)
		#print("****** "+text5)
		print("arabe "+text6)
		if L !=[] or len(text6)==5:
			result=tt+"TN"+kk
		elif kk.find(" ") !=0:
			result=tt+"TN"+kk[kk.find(" ")+1:]
		else:
			result=tt+"TN"+kk
			
		#print("**right part** "+kk)
		#print("**left part** "+tt)
		#print("**result** "+result)
		matricule.append(result)
		cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
		
	# Cropped = cv.resize(frame,(right, bottom))
	
		label = '%.2f' % conf

		# Get the label for the class name and its confidence
		if classes:
			assert(classId < len(classes))
			label = '%s:%s' % (classes[classId], label)

		#Display the label at the top of the bounding box
		labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		top = max(top, labelSize[1])
		cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
		#cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
		cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
		return result
	# Remove the bounding boxes with low confidence using non-maxima suppression
	def convert_rgb_to_names(rgb_tuple):
		# a dictionary of all the hex and their respective names in css3
		css3_db = CSS3_HEX_TO_NAMES#css3_hex_to_names
		names = []
		rgb_values = []    
		for color_hex, color_name in css3_db.items():
			names.append(color_name)
			rgb_values.append(hex_to_rgb(color_hex))
		
		kdt_db = KDTree(rgb_values)    
		distance, index = kdt_db.query(rgb_tuple)
		return names[index]

	coco_names = [
		'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
		'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
		'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
		'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
		'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
		'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
		'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
		'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
		'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
		'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
		
	
	]
	COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

	# read an image from the internet
	url = "https://img-4.linternaute.com/JLKw5PYU8I_JP9POLIQAqBdhLMg=/1500x/smart/fac6d59eb91a44e886947968af8ecb2e/ccmcms-linternaute/21413766.jpg"
	#response = requests.get(url)
	#Cropped=frame[top:bottom, left:right]
	#Cropped = cv.resize(Cropped,(200,50))
	image = Image.open(imagePath).convert("RGB")

	# create a retinanet inference model
	model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, score_thresh=0.3)
	model.eval()

	# predict detections in the input image
	image_as_tensor = transforms.Compose([transforms.ToTensor(), ])(image)
	outputs = model(image_as_tensor.unsqueeze(0))

	# post-process the detections ( filter them out by score )
	detection_threshold = 0.5
	pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
	pred_scores = outputs[0]['scores'].detach().cpu().numpy()
	pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
	boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
	classes = pred_classes
	labels = outputs[0]['labels']

	objNature=[]
	matricule=[]
	objcouleur=[]
	# draw predictions
	image = cv.cvtColor(np.asarray(image), cv.COLOR_BGR2RGB)
	for i, box in enumerate(boxes):
		color = COLORS[labels[i]]
		Cropped=image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
		
		cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
		cv.putText(image, classes[i] , (int(box[0]), int(box[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
					lineType=cv.LINE_AA)
		if classes[i] =="car":
			cv.imwrite("couleur.jpg", Cropped)
			objNature.append(classes[i])  
			Cropped1=Image.open("couleur.jpg")
			img = Cropped1.copy()
			#img=Image.fromarray(img)
			img.convert("RGB")
			height = np.size(img, 0)
			width = np.size(img, 1)
			img = img.resize((height, width), resample=0)
		
			#Cropped2=Image.open(img)
			Cropped2=cv.cvtColor(np.asarray(img), cv.COLOR_BGR2RGB)
		#cv2.imshow('Image 3', Cropped2)
			s=(2*width)/3  
			dominant_color = img.getpixel((height/2, (width/2)+5))
			named_color = convert_rgb_to_names(dominant_color)
			objcouleur.append(named_color)
			#print("couleur dominant: "+named_color)           
			#print("this is "+ classes[i] + "nombre de voiture " +str(i))
	#print("nombre de voiture " +str(i+1)) 



	def postprocess(frame, outs):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]

		classIds = []
		confidences = []
		boxes = []
		# Scan through all the bounding boxes output from the network and keep only the
		# ones with high confidence scores. Assign the box's class label as the class with the highest score.
		classIds = []
		confidences = []
		boxes = []
		for out in outs:
			print("out.shape : ", out.shape)
			for detection in out:
				#if detection[4]>0.001:
				scores = detection[5:]
				classId = np.argmax(scores)
				#if scores[classId]>confThreshold:
				confidence = scores[classId]
				if detection[4]>confThreshold:
					print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
					print(detection)
				if confidence > confThreshold:
					center_x = int(detection[0] * frameWidth)
					center_y = int(detection[1] * frameHeight)
					width = int(detection[2] * frameWidth)
					height = int(detection[3] * frameHeight)
					left = int(center_x - width / 2)
					top = int(center_y - height / 2)
					classIds.append(classId)
					confidences.append(float(confidence))
					boxes.append([left, top, width, height])

		# Perform non maximum suppression to eliminate redundant overlapping boxes with
		# lower confidences.
		indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
		for i in indices:
			i = i[0]
			box = boxes[i]
			left = box[0]
			top = box[1]
			width = box[2]
			height = box[3]
			drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

	# Process inputs
	winName = 'Deep learning object detection in OpenCV'
	cv.namedWindow(winName, cv.WINDOW_NORMAL)

	outputFile = "yolo_out_py.avi"
	if (imagePath):
		# Open the image file
		if not os.path.isfile(imagePath):
			print("Input image file ", imagePath, " doesn't exist")
			sys.exit(1)
		cap = cv.VideoCapture(imagePath)
		outputFile ='output\\'+imagePath+'_yolo_out_py.jpg'
	elif (args.video):
		# Open the video file
		if not os.path.isfile(args.video):
			print("Input video file ", args.video, " doesn't exist")
			sys.exit(1)
		cap = cv.VideoCapture(args.video)
		outputFile = args.video[:-4]+'_yolo_out_py.avi'
	else:
		# Webcam input
		cap = cv.VideoCapture(0)

	# Get the video writer initialized to save the output video
	if (not imagePath):
		vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

	while cv.waitKey(1) < 0:

		# get frame from the video
		hasFrame, frame = cap.read()

		# Stop the program if reached end of video
		if not hasFrame:
			print("Done processing !!!")
			print("Output file is stored as ", outputFile)
			cv.waitKey(3000)
			break

		# Create a 4D blob from a frame.
		blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

		# Sets the input to the network
		net.setInput(blob)

		# Runs the forward pass to get output of the output layers
		outs = net.forward(getOutputsNames(net))

		# Remove the bounding boxes with low confidence
		postprocess(frame, outs)
		
		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
		#cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

		# Write the frame with the detection boxes
		if (imagePath):
			cv.imwrite(outputFile, frame.astype(np.uint8));
		else:
			vid_writer.write(frame.astype(np.uint8))
	print(objNature[0]+" "+matricule[0])
	date = datetime.datetime.now()


  