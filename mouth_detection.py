import numpy as np
import cv2
from math import copysign, log10

face_cascade = cv2.CascadeClassifier('Classifiers\\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('Classifiers\\Mouth.xml')

cv2.namedWindow('thresh')

#MOUTH_OPEN = [3.00827653, 9.84436131, 11.31427271, 11.66784116, 23.23944105, -16.65914565, -23.41328958]
#MOUTH_CLOSED = [3.01481441, 10.13538961, 11.78238785, 12.82624992, -25.32969439, 18.22691494, 25.24138952]

MOUTH_CLOSED = cv2.imread('mouth_closed.png', cv2.IMREAD_GRAYSCALE)
MOUTH_OPEN = cv2.imread('mouth_open.png', cv2.IMREAD_GRAYSCALE)

last_mouth = MOUTH_CLOSED

def run_camera():
    global last_mouth
    cap = cv2.VideoCapture(0)
    
    while (True):

        # Read Image From Webcam
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

        mouth = (-1, -1, -1, -1) #(mx,my,mw,mh)
        mouth_open = False
        
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #roi_gray = gray[int(y+h*2/3):y+h, int(x+w/3):int(x+w*2/3)]
            #roi_color = frame[int(y+h*2/3):y+h, int(x+w/3):int(x+w*2/3)]
            roi_gray = gray[int(y+h*2/3):int(y+h*2/3 + 97), int(x+w/3):int(x+w/3 + 97)]
            roi_color = frame[int(y+h*2/3):int(y+h*2/3 + 97), int(x+w/3):int(x+w/3 + 97)]
#
##            try:
##                mouth_delta = cv2.absdiff(last_mouth, roi_gray).mean(0).mean()
##                print (mouth_delta)
##            except:
##                pass
##
##            last_mouth = roi_gray

            mouths = mouth_cascade.detectMultiScale(roi_gray)
            mouth_open = get_mouth_state(roi_gray, roi_color)
            if (len(mouths) == 0):
                break
            elif (len(mouths) > 1):
              mouth = mouths[0]  
            else:
                mouth = mouths[0]
                for m in mouths:
                    if (m[1] > mouth[1]):
                        mouth = m

        if (mouth[0] != -1):
            (mx,my,mw,mh) = mouth
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,255,0),2)

        cv2.imshow('faces', cv2.flip(frame, 1))
        cv2.waitKey(1)

        # Escape Sequence
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_mouth_state(roi_mouth, roi_color):
    global img_saved
    #ret, thresh = cv2.threshold(roi_mouth, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(roi_mouth,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(roi_color, contours, 0, (0, 255, 0), 4)
    moments = cv2.moments(thresh)
    huMoments = cv2.HuMoments(moments)
    for i in range(0,7):
        huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
       # print (huMoments[i])

    #if (huMoments[4] > 0):
       # print ('Mouth Open')
    #else:
        #print ('Mouth Closed')
        
   # dist_open = cv2.matchShapes(roi_mouth, MOUTH_OPEN, cv2.CONTOURS_MATCH_I1, 0)
    #dist_closed = cv2.matchShapes(roi_mouth, MOUTH_CLOSED, cv2.CONTOURS_MATCH_I1, 0)
    #print (' ')
   # print ('Dist Open: {}'.format(dist_open))
   # print('Dist Closed: {}'.format(dist_closed))
    #cv2.drawContours(roi_color, [c], 0, (0, 255, 0), 4)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(200)
    return False

def distance(m1, m2):
    diff = 0
    for i in range(len(m1)):
        diff += abs(m1[i] - m2[i])
    return diff

if __name__=="__main__":
    run_camera()
