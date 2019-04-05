import numpy as np
import cv2



CIRCLE_DIAMETER = 3 # Diameter of Colored Circles in Inches

GREEN_THRESHOLD = ((50, 127, 64), (90, 255, 255)) # Low and High HSV Threshold for Green
RED_THRESHOLD = ((0, 127, 64), (6, 255, 255)) # Low and High HSV Threshold for Red


def run_camera():
    cap = cv2.VideoCapture(1)
    
    while (True):

        # Read Image From Webcam
        ret, frame = cap.read()

        # Image Preprocessing
        frame = cv2.flip(frame, 1)
        img_out = frame.copy()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filter Colors
        green_mask = cv2.inRange(hsv, GREEN_THRESHOLD[0], GREEN_THRESHOLD[1])
        red_mask = cv2.inRange(hsv, RED_THRESHOLD[0], RED_THRESHOLD[1])

        kp_green = largest_keypoint(detector.detect(green_mask))
        kp_red = largest_keypoint(detector.detect(red_mask))

        # Draw Circles on Detected Blobs
        if (kp_green != None):
            cv2.circle(img_out, (int(kp_green.pt[0]), int(kp_green.pt[1])), int(kp_green.size/2+.5), (0, 0, 255), 4)
        
        if (kp_red != None):
            cv2.circle(img_out, (int(kp_red.pt[0]), int(kp_red.pt[1])), int(kp_red.size/2+.5), (0, 255, 0), 4)


        note = get_note(kp_green, kp_red)

        cv2.imshow('Frame', img_out)
        #cv2.imshow('Red', red_mask)

        # Escape Sequence
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_note(kp_green, kp_red):
    

# Find Largest Detected Blob
def largest_keypoint(keypoints):
    if (len(keypoints) == 0):
        return None
    elif (len(keypoints) == 1):
        return keypoints[0]
    else:
        largest_kp = keypoints[0]
        for kp in keypoints:
            if kp.size > largest_kp.size:
                largest_kp = kp
        return kp

# Setup Blob Detector Parameters
def setup_blob_detector():
    global detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 255
    
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 50000

    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)



if __name__=="__main__":
    setup_blob_detector()
    run_camera()
