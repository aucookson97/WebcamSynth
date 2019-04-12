import numpy as np
import cv2
import rtmidi

MIDDLE_NOTE = 60 # What Note to Play When Green and Red are at the same Y
INCHES_PER_NOTE = 1 # Note Resolution in Inches
VEL_PER_INCHES = 5 # Velocity Resolution in Inches
THRESHOLD_DISTANCE = 20 # Split Distance Away that defines note off threshold

CIRCLE_DIAMETER = 3 # Diameter of Colored Circles in Inches

GREEN_THRESHOLD = ((50, 127, 64), (90, 255, 255)) # Low and High HSV Threshold for Green
RED_THRESHOLD = ((0, 127, 64), (6, 255, 255)) # Low and High HSV Threshold for Red

threshold_velocity = int(127 - THRESHOLD_DISTANCE * VEL_PER_INCHES)

last_note = MIDDLE_NOTE

# UI Params
LINE_LENGTH = 20

def run_camera():
    cap = cv2.VideoCapture(0)
    
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

        if (kp_green != None and kp_red != None):
            note = get_note(kp_green, kp_red)
            draw_ui(img_out, kp_green, kp_red)

        if (kp_red == None or kp_green == None):
            note_off = [0x80, last_note, 0]
            midiout.send_message(note_off)

        cv2.imshow('Frame', img_out)
        #cv2.imshow('Red', red_mask)

        # Escape Sequence
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_note(kp_green, kp_red):
    global last_note
    relative_loc_pixels_pitch = kp_red.pt[0] - kp_green.pt[0]
    relative_loc_pixels_velocity = kp_red.pt[1] - kp_green.pt[1]

    ppi = get_ppi(kp_green, kp_red)

    relative_loc_inches_pitch = relative_loc_pixels_pitch / ppi
    relative_loc_inches_velocity = abs(relative_loc_pixels_velocity / ppi)


    pitch = int(MIDDLE_NOTE + (relative_loc_inches_pitch / INCHES_PER_NOTE) + .5)
    velocity = int(127 - relative_loc_inches_velocity * VEL_PER_INCHES  + .5)

    if (velocity <= threshold_velocity):
        velocity = 0


    if (pitch != last_note):
        note_off = [0x80, last_note, 0]
        note_on = [0x90, pitch, velocity]
        midiout.send_message(note_on)
        midiout.send_message(note_off)
        print (pitch, velocity)
        last_note = pitch

    return 1

def draw_ui(img_out, kp_green, kp_red):
    ppi = get_ppi(kp_green, kp_red)

    # Draw Note OFF Threshold Line
    loc_x = int(kp_green.pt[0] - ppi * THRESHOLD_DISTANCE)

##    cv2.line(img_out, (loc_x, int(kp_red.pt[1] - LINE_LENGTH * 4)), (loc_x, int(kp_red.pt[1] + LINE_LENGTH * 4)),
##                 (255, 85, 0), 4)

##    # Draw Higher Note Lines
##    loc_y = int(kp_red.pt[1])
##    current_note = MIDDLE_NOTE
##    while (loc_y > 0):
##        if (current_note % 12 == 0): # Octaves are Longer
##            length = LINE_LENGTH * 2
##        else:
##            length = LINE_LENGTH
##        pt1 = (kp_green.pt[0] - length, loc_y)
##
##        cv2.line(img_out, (int(kp_green.pt[0] - length / 2), loc_y), (int(kp_green.pt[0] + length / 2), loc_y),
##                 (255, 85, 0), 4)
##
##        loc_y -= ppi * INCHES_PER_NOTE
##        current_note += 1
##
##    # Draw Lower Note Lines
##    loc_y = int(kp_red.pt[1])
##    current_note = MIDDLE_NOTE
##    while (loc_y < img_out.shape[0]):
##        if (current_note % 12 == 0): # Octaves are Longer
##            length = LINE_LENGTH * 2
##        else:
##            length = LINE_LENGTH
##        pt1 = (kp_green.pt[0] - length, loc_y)
##
##        cv2.line(img_out, (int(kp_green.pt[0] - length / 2), loc_y), (int(kp_green.pt[0] + length / 2), loc_y),
##                 (255, 0, 0), 4)
##
##        loc_y += ppi * INCHES_PER_NOTE
##        current_note -= 1


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

# Calculate Pixels Per Inch using Size of Circles
def get_ppi(kp_green, kp_red):
    return int(max(kp_green.size, kp_red.size) / CIRCLE_DIAMETER + .5)

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

def setup_midiout():
    global midiout
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()

    if available_ports:
        midiout.open_port(1)
    else:
        midiout.open_virtual_port("Python Virtual Out")

def all_notes_off():
    pass

if __name__=="__main__":
    setup_midiout()
    setup_blob_detector()
    run_camera()
