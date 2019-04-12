import cv2
import numpy as np

TARGET_SIZE = 2.75

template = cv2.cvtColor(cv2.imread("target_template.png"), cv2.COLOR_BGR2GRAY)
width, height = template.shape[::-1]

def run_camera():
    cap = cv2.VideoCapture(0)
    
    while (True):

        # Read Image From Webcam
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc

        bottom_right = (top_left[0] + width, top_left[1] + height)

        cv2.rectangle(frame, top_left, bottom_right, 255, 4)

        cv2.imshow("Template", frame)
        cv2.waitKey(1)

        # Escape Sequence
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__=="__main__":
    run_camera()
