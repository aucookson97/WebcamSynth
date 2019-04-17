import numpy as np
import cv2
import rtmidi

MIDDLE_NOTE = 74 # What Note to Play When Green and Red are at the same Y
INCHES_PER_NOTE = 1.5 # Note Resolution in Inches
VEL_PER_INCHES = 5 # Velocity Resolution in Inches
THRESHOLD_DISTANCE = 20 # Split Distance Away that defines note off threshold

CIRCLE_DIAMETER = 3 # Diameter of Colored Circles in Inches

GREEN_THRESHOLD = ((50, 100, 64), (90, 255, 255)) # Low and High HSV Threshold for Green
RED_THRESHOLD = ((0, 100, 64), (10, 255, 255)) # Low and High HSV Threshold for Red

threshold_velocity = int(127 - THRESHOLD_DISTANCE * VEL_PER_INCHES)

last_cc_val_x = 0
last_cc_val_y = 0

# UI Params
LINE_LENGTH = 80
LINE_COLOR = (100, 12, 90)
LINE_WIDTH = 6

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

def run_camera():
    global last_cc_val_x, last_cc_val_y
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
            #draw_ui(img_out, kp_green, kp_red)


        if (kp_green != None and kp_red != None):
            cc = get_cc(kp_green, kp_red)
            if (cc[0] != last_cc_val_x):
                midiout.send_message((0xb0, 0x10, cc[0]))
                last_cc_val_x = cc[0]
           ## print (cc[1])

            if (cc[1] != last_cc_val_y):
                midiout.send_message((0xb0, 0x11, cc[1]))
                last_cc_val_y = cc[0]

        cv2.imshow('Frame', img_out)
        #cv2.imshow('Red', red_mask)

        # Escape Sequence
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_cc(kp_green, kp_red):
    relative_loc_pixels_x =  kp_green.pt[0] - kp_red.pt[0]
    relative_loc_pixels_y = -(kp_green.pt[1] - kp_red.pt[1])
    
    ppi = get_ppi(kp_green, kp_red)
    cc_per_inch_x = 127.0 / (20 - 4)
    cc_per_inch_y = 127.0 / (24)

    cc_val_x = int(max(min(cc_per_inch_x * (relative_loc_pixels_x / ppi - 4), 127), 0))
    cc_val_y = int(max(min(cc_per_inch_y * (12 + relative_loc_pixels_y / ppi), 100), 0))
    
    cc = (cc_val_x, cc_val_y)
    return cc



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
    #loc_x = int(kp_green.pt[0] - ppi * THRESHOLD_DISTANCE)

##    cv2.line(img_out, (loc_x, int(kp_red.pt[1] - LINE_LENGTH * 4)), (loc_x, int(kp_red.pt[1] + LINE_LENGTH * 4)),
##                 (255, 85, 0), 4)

    loc_x = int(kp_red.pt[0])
    loc_y = int(kp_red.pt[1])
    current_note = MIDDLE_NOTE
    while (loc_x < img_out.shape[1]):
        loc_x += ppi * INCHES_PER_NOTE
        current_note -= 1

        if (current_note == 64):
            length = LINE_LENGTH * 2
        else:
            length = LINE_LENGTH
        text = NOTE_NAMES[current_note % 12]
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        cv2.line(img_out, (int(loc_x), int(loc_y - length/2)), (int(loc_x), int(loc_y + length/2)), LINE_COLOR, LINE_WIDTH)
        cv2.putText(img_out, text, (int(loc_x - textSize[0][0] / 2), int(loc_y + LINE_LENGTH + 10 + textSize[0][1])),
                    cv2.FONT_HERSHEY_DUPLEX, 1, LINE_COLOR, 2)
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
    return int(kp_red.size / CIRCLE_DIAMETER + .5)

# Setup Blob Detector Parameters
def setup_blob_detector():
    global detector
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 255
    
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 50000

    params.filterByInertia = True
    params.minInertiaRatio = .5
    params.maxInertiaRatio = 1

    params.filterByCircularity = False
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
