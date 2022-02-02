import cv2 as cv
import pytesseract
from pytesseract import image_to_string
from imutils import paths
import argparse
import imutils

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
#cv.imshow('result 4',result4)
    text33 = pytesseract.image_to_string(img, config='-l eng --oem 1 --psm 8')
#print(pytesseract.image_to_string(imagem,config ='-l eng --psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
    print("result de "+str(s)+" :" +text33)




cv.waitKey(0)