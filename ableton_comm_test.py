import time
import rtmidi
import cv2

midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()

if available_ports:
    midiout.open_port(0)
else:
    midiout.open_virtual_port("Python Virtual Out")

note_on = [0x90, 60, 112]
note_off = [0x80, 60, 0]

midiout.send_message(note_on)
time.sleep(.5)
midiout.send_message(note_off)

del midiout
