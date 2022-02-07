import cv2 as cv
import pytesseract
from pytesseract import image_to_string
from imutils import paths
import argparse
import imutils
from ArabicOcr import arabicocr
ap = argparse.ArgumentParser()
ap.add_argument('--i', '-input', required=True,help="path to input directory of images", type=str)
#parser.add_argument('--video', help='Path to video file.')
args = ap.parse_args()
imagePaths = sorted(list(paths.list_images(args.i)))
s=0
for imagePath in imagePaths:
    s=s+1
    img = cv.imread(imagePath)
    img =  cv.resize(img ,(200,30))
    t=cv.imwrite("test2.jpg",img)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)


    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    _, result = cv.threshold(img, 80, 255, cv.THRESH_BINARY)

    adaptive = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 81, 4)

#cv.imshow('Image', img)
#cv.imshow('Result', result)
#cv.imshow('Result ADAPTIVE', adaptive)

    cv.imwrite('resultat1.jpg', adaptive)
    cv.imwrite('resultat2.jpg', result)
    _, result4 = cv.threshold(result, 80, 255, cv.THRESH_BINARY_INV)
    cv.imwrite('resultat4.jpg', result4)
    imagem = cv.bitwise_not(result)
    cv.imwrite("resultat3.jpg", imagem)
    rectKern = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))
    blackhat = cv.morphologyEx(result4, cv.MORPH_BLACKHAT, rectKern)
    cv.imwrite("resultat5.jpg", blackhat)

    
    roi = cv.threshold(img, 0, 255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    cv.imwrite("resultat6.jpg", roi)
    image_path='temp2.jpg'
    out_image='out.jpg'
    text34 = pytesseract.image_to_string(image_path, config='-l eng --psm 11')
    #print("text34= "+text34)
    text33 = arabicocr.arabic_ocr(image_path,out_image)
    words=[]
    tab=['A','B','C','D','E','F','J','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','Z','a','b','c','d','e','f','j','h','i','g','k','l','m','n','o','p','q','r','s','t','w','z']
    for i in range(len(text33)):	
	    word=text33[i][1]
	    print("word="+word)
	    word.replace('/','').replace('٠','')
	    words.append(word)
			    
	    
	
			
#print(pytesseract.image_to_string(imagem,config ='-l eng --psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    if len(words)==1:
		    k = "".join(words)
		    k=k.replace('.','').replace('|','').replace('/','4').replace(']','')
		    pm1=pytesseract.image_to_string(image_path, config='-l ara --oem 1 --psm 3')
		    if k.find("و") != -1 and k.find("و") != len(k)-1 or k.find("ن") != -1 and k.find("ن") != len(k)-1:
			    print("matricule1 =")
			    if k.find("L") == -1:
				    print(k[len(k)-3:]+"TN"+k[0:4])
			    else:
					#image_path1= image2.name+'.jpg'
				    results1=arabicocr.arabic_ocr(image_path,out_image)
				    print(k[len(k)-3:]+"TN"+k[0:4])
				    #print(k[len(k)-3:]+"TN"+k[0:4])
					#print(results1)


					
				
    if len(words)==2:
					

			
	    k1 = "".join(words[0])
	    k2 = "".join(words[1])
	    pm2=pytesseract.image_to_string(imagePath, config='-l ara --oem 1 --psm 3')
				#print("matricule =")
				#print(k1+"TN"+k2[0:4])
				
				
	    if k2.find("ن") == -1 and pm2.find('ن') !=-1 and pm2.find('ن') !=len(pm2)-1 or pm2.find('ت') !=-1 and pm2.find('ت') !=len(pm2)-1 or len(k2)>4:
			    print("matricule2=")
			    k1=k1.replace('&','6').replace('ن','').replace(' ','').replace('/','').replace('٠','')
					#print(pytesseract.image_to_string(Cropped, config='-l ara --oem 1 --psm 3'))
			    print(k1[0:4]+"TN"+k2[0:4])
		 
			   			
	    if k1[0]==5 and len(k1[0:4])>2:
			    print(k2[0:4]+"TN"+k1[0:4])		
	    
	
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


cv.waitKey(0)