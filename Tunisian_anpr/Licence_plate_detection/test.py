from ArabicOcr import arabicocr
image_path='filename3.jpg'
out_image='out.jpg'
results=arabicocr.arabic_ocr(image_path,out_image)
print(results)
words=[]

	
for i in range(len(results)):	
		word=results[i][1]
		print("word="+word)
		words.append(word)

	
	   	
#del words[0:2]
		
with open ('file.txt','w',encoding='utf-8')as myfile:
		myfile.write(str(words))
import cv2
img = cv2.imread('out.jpg', cv2.IMREAD_UNCHANGED)
cv2.imshow("arabic ocr",img)
if len(words)==1:
	k = "".join(words)
	k=k.replace('.','').replace('|','')
	
	if k.find("و") != -1 and k.find("و") != len(k)-1 or k.find("ن") != -1 and k.find("ن") != len(k)-1:
	    print("matricule =")
	    print(k[len(k)-3:]+"TN"+k[0:4])
if len(words)==2:
	k1 = "".join(words[0])
	k2 = "".join(words[1])
    
	print("matricule =")
	#print(k1+"TN"+k2[0:4])
	if k1.find("ن") != -1:
		print("matricule")
		print(k1+"TN"+k2[0:4])
	if k2.find("ن") != -1 and k2.find("و")!=-1 and k2.find("ت")!=-1 and :
		print("matricule")
		print(k1[0:4]+"TN"+k2[0:4])	
if len(words)==3:
	k1 = "".join(words[0])
	k2 = "".join(words[2])
	k3 = "".join(words[1])
	if k3 in ['تونس'] or k3 in ['لوذ']:
	   print("matricule =")
	   print(k1+"TN"+k2)
if len(words)>3:
	for i in range(len(words)):
	    ki = "".join(words[i])

	    if ki in ['تونس']:
	        ks = "".join(words[i-1])
	        ks1 = "".join(words[i-2])
	        print("matricule =")
	        print(ks1+"TN"+ks)	   				
cv2.waitKey(0)   